import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.llm.provider import create_chat_model
from src.llm.response import extract_text


class RAGEvaluator:

    def __init__(
        self,
        model="qwen2.5:7b",
        provider="ollama",
        api_key=None,
    ):
        self.provider = provider.strip().lower()

        if self.provider == "ollama":


            # Primary evaluator
            # These are the frozen benchmark settings.

            self.llm = ChatOllama(
                model=model,
                temperature=0,
                seed=42,
                format="json",
                num_predict=256,
                repeat_penalty=1.1
            )

            # First fallback
            # Used only if the primary evaluator hits
            # Ollama's token-repeat abort.

            self.fallback_llm = ChatOllama(
                model=model,
                temperature=0.2,
                seed=42,
                format="json",
                num_predict=128,
                repeat_penalty=1.25
            )

            # Second fallback
            # Used only if BOTH:
            # 1. primary evaluator fails with token repeat
            # 2. first fallback also fails with token repeat
            # This is paired with a more compact
            # faithfulness prompt.

            self.second_fallback_llm = ChatOllama(
                model=model,
                temperature=0.3,
                seed=42,
                format="json",
                num_predict=96,
                repeat_penalty=1.3
            )



        else:
            self.llm = create_chat_model(
                provider=self.provider,
                model=model,
                api_key=api_key,
                temperature=0,
            )

            # Ollama-specific decoding fallbacks
            # are not needed for cloud providers.
            self.fallback_llm = self.llm
            self.second_fallback_llm = self.llm

        # Recovery diagnostics

        self.fallback_count = 0

        self.second_fallback_count = 0

        self.fallback_events = []

        # Retrieval relevance evaluator

        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are evaluating one retrieved passage from a RAG system.

The passage does not need to completely answer the question by itself.

It only needs to provide useful evidence or information that could help
the final answer.

Use this scale:

0 = unrelated to the question

1 = related background or partially useful information

2 = directly useful evidence for answering the question

For example, if the question asks how to catch Python errors,
passages explaining try/except, exception handling, catching exceptions,
or examples of exception-handling code should receive a score of 2.

Do not penalize a passage just because it does not contain the entire answer.

Return JSON only:

{{
    "score": 0,
    "reason": "short explanation"
}}
"""
            ),
            (
                "human",
                """
Question:

{query}

Retrieved passage:

{passage}
"""
            )
        ])


        # Answer relevance evaluator

        self.answer_relevance_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
Evaluate only whether the generated answer addresses the user's question.

Do not evaluate:
- whether the answer is exhaustive
- whether it includes best practices
- whether another detail could have been added
- whether the answer is supported by retrieved context
- writing style or verbosity

Judge only whether it provides the information or action the user asked for.

Use this scale:

0 = unrelated to the question or answers a different question

1 = discusses the correct topic and provides useful related information,
but does not actually provide the specific information or action requested

2 = directly provides the information or action requested

A short answer can receive 2.

An answer does not need to be comprehensive to receive 2.

Do not reduce a score from 2 to 1 merely because additional details,
examples, best practices, caveats, or explanation could have been included.

Example:

Question:

How do I convert a string to an integer?

Answer:

Use int() on the string.

Score:

2

The answer is brief but directly provides the requested method.

Example:

Question:

How do I convert a string to an integer?

Answer:

Python supports several numeric data types.

Score:

1

The answer discusses a related topic but does not explain how to perform
the requested conversion.

Example:

Question:

How do I convert a string to an integer?

Answer:

Lists can contain multiple values.

Score:

0

The answer is unrelated.

A response should receive a score of 2 when it directly states the
definition, explanation, method, or action requested by the question,
even if the wording is brief, informal, or different from an ideal
textbook answer.

For definition questions, directly stating what the requested concept
means is sufficient for a score of 2.

For "how" questions, providing a concrete method or action that answers
the question is sufficient for a score of 2.

Use score 1 only when the reader would still need additional information
to know the specific definition, method, or action that was requested.

Return only a valid JSON object containing exactly two fields:

- score: an integer (0, 1, or 2) based on the evaluation scale above
- reason: one specific sentence explaining why the answer deserves that score

Evaluate the actual question and generated answer.
Do not use a placeholder score or explanation.
Do not include Markdown formatting or text outside the JSON object.
"""
            ),
            (
                "human",
                """
Question:

{query}

Generated answer:

{answer}
"""
            )
        ])


        # Main faithfulness evaluator

        self.faithfulness_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
Evaluate whether the factual claims in a generated RAG answer
are supported by the retrieved context.

Identify the important factual claims in the answer.

Put claims that are supported by the retrieved context in
"supported_claims".

Put claims that are not supported by the retrieved context in
"unsupported_claims".

A claim is supported when the retrieved context clearly provides
the same fact or meaning.

Do not mark the whole answer as unfaithful just because one claim
is unsupported.

Return JSON only:

{{
    "supported_claims": [
        "supported factual claim"
    ],
    "unsupported_claims": [
        "unsupported factual claim"
    ],
    "reason": "short overall explanation"
}}
"""
            ),
            (
                "human",
                """
Retrieved context:

{context}

Generated answer:

{answer}
"""
            )
        ])


        # Compact faithfulness recovery prompt
        # Same evaluation rule, but forces a short output.
        # This is ONLY reached after both normal decoding
        # attempts fail with a token-repeat error.

        self.compact_faithfulness_prompt = (
            ChatPromptTemplate.from_messages([
                (
                    "system",
                    """
Evaluate whether the important factual claims in the generated answer
are supported by the retrieved context.

Use the same rule:

A claim is supported when the retrieved context clearly provides
the same fact or meaning.

Keep the response very short.

Return at most 4 supported claims and at most 4 unsupported claims.

Each claim must be no more than 12 words.

Do not repeat claims.

Return JSON only:

{{
    "supported_claims": [],
    "unsupported_claims": [],
    "reason": "brief explanation"
}}
"""
                ),
                (
                    "human",
                    """
Retrieved context:

{context}

Generated answer:

{answer}
"""
                )
            ])
        )


        # --------------------------------------------------
        # Primary chains
        # --------------------------------------------------

        self.retrieval_chain = (
            self.retrieval_prompt
            | self.llm
        )


        self.answer_relevance_chain = (
            self.answer_relevance_prompt
            | self.llm
        )


        self.faithfulness_chain = (
            self.faithfulness_prompt
            | self.llm
        )


        # --------------------------------------------------
        # First fallback chains
        #
        # Same prompts, slightly safer decoding.
        # --------------------------------------------------

        self.retrieval_fallback_chain = (
            self.retrieval_prompt
            | self.fallback_llm
        )


        self.answer_relevance_fallback_chain = (
            self.answer_relevance_prompt
            | self.fallback_llm
        )


        self.faithfulness_fallback_chain = (
            self.faithfulness_prompt
            | self.fallback_llm
        )


        # --------------------------------------------------
        # Second faithfulness fallback chain
        # --------------------------------------------------

        self.faithfulness_second_fallback_chain = (
            self.compact_faithfulness_prompt
            | self.second_fallback_llm
        )


    # --------------------------------------------------
    # Detect the specific Ollama failure
    # --------------------------------------------------

    def _is_repeat_error(
        self,
        error
    ):

        return (
            "token repeat limit reached"
            in str(error).lower()
        )


    # --------------------------------------------------
    # Safe evaluator invocation
    # --------------------------------------------------

    def _invoke_with_fallback(
        self,
        primary_chain,
        fallback_chain,
        inputs,
        stage,
        second_fallback_chain=None
    ):

        try:

            return primary_chain.invoke(
                inputs
            )


        except Exception as error:

            if not self._is_repeat_error(
                error
            ):
                raise


            self.fallback_count += 1


            self.fallback_events.append(
                f"{stage}:level_1"
            )


            print(
                "\n[Evaluator fallback]"
            )


            print(
                f"Stage: {stage}"
            )


            print(
                "Ollama hit the token-repeat limit."
            )


            print(
                "Retrying with first-level "
                "bounded decoding..."
            )


            try:

                return fallback_chain.invoke(
                    inputs
                )


            except Exception as fallback_error:

                # Any error other than the exact
                # repetition failure should still
                # propagate normally.

                if not self._is_repeat_error(
                    fallback_error
                ):
                    raise


                # Only faithfulness currently has
                # a second-level recovery chain.

                if second_fallback_chain is None:
                    raise


                self.second_fallback_count += 1


                self.fallback_events.append(
                    f"{stage}:level_2"
                )


                print(
                    "\n[Evaluator second fallback]"
                )


                print(
                    f"Stage: {stage}"
                )


                print(
                    "First fallback also hit "
                    "the token-repeat limit."
                )


                print(
                    "Retrying with compact "
                    "evaluation output..."
                )


                return (
                    second_fallback_chain.invoke(
                        inputs
                    )
                )


    # --------------------------------------------------
    # Retrieval evaluation
    # --------------------------------------------------

    def evaluate_retrieval(
        self,
        query,
        results
    ):

        chunk_scores = []


        for result in results:

            response = (
                self._invoke_with_fallback(
                    self.retrieval_chain,
                    self.retrieval_fallback_chain,
                    {
                        "query":
                            query,

                        "passage":
                            result["text"]
                    },
                    stage=
                        "retrieval_relevance"
                )
            )


            try:

                evaluation = json.loads(
                    extract_text(response.content)
                )


                score = int(
                    evaluation.get(
                        "score",
                        0
                    )
                )


                score = max(
                    0,
                    min(
                        score,
                        2
                    )
                )


            except (
                json.JSONDecodeError,
                ValueError,
                TypeError
            ):

                score = 0


                evaluation = {
                    "reason":
                        "Evaluator output "
                        "could not be parsed."
                }


            chunk_scores.append({
                "page":
                    result["page"],

                "score":
                    score,

                "reason":
                    evaluation.get(
                        "reason",
                        ""
                    )
            })


        if not chunk_scores:

            return {
                "retrieval_relevance":
                    0.0,

                "relevant_fraction":
                    0.0,

                "direct_fraction":
                    0.0,

                "chunks":
                    []
            }


        scores = [
            item["score"]
            for item
            in chunk_scores
        ]


        relevant_fraction = (
            sum(
                score >= 1
                for score
                in scores
            )
            / len(scores)
        )


        direct_fraction = (
            sum(
                score == 2
                for score
                in scores
            )
            / len(scores)
        )


        top_scores = sorted(
            scores,
            reverse=True
        )[:3]


        retrieval_relevance = (
            sum(
                top_scores
            )
            / (
                len(top_scores)
                * 2
            )
        )


        return {
            "retrieval_relevance":
                retrieval_relevance,

            "relevant_fraction":
                relevant_fraction,

            "direct_fraction":
                direct_fraction,

            "chunks":
                chunk_scores
        }


    # --------------------------------------------------
    # Answer evaluation
    # --------------------------------------------------

    def evaluate_answer(
        self,
        query,
        results,
        answer
    ):

        # ----------------------------------------------
        # Answer relevance
        # ----------------------------------------------

        relevance_response = (
            self._invoke_with_fallback(
                self.answer_relevance_chain,
                self.answer_relevance_fallback_chain,
                {
                    "query":
                        query,

                    "answer":
                        answer
                },
                stage=
                    "answer_relevance"
            )
        )


        try:

            relevance_data = json.loads(
                extract_text(relevance_response.content)
            )


            relevance_score = int(
                relevance_data.get(
                    "score",
                    0
                )
            )


            relevance_score = max(
                0,
                min(
                    relevance_score,
                    2
                )
            )


        except (
            json.JSONDecodeError,
            ValueError,
            TypeError
        ):

            relevance_score = 0


            relevance_data = {
                "reason":
                    "Answer relevance output "
                    "could not be parsed."
            }


        # ----------------------------------------------
        # Build retrieved context
        # ----------------------------------------------

        context = "\n\n".join(
            (
                f"Page "
                f"{result['page']}:\n"
                f"{result['text']}"
            )
            for result
            in results
        )


        # ----------------------------------------------
        # Faithfulness
        #
        # Primary
        # → first fallback
        # → compact second fallback
        # ----------------------------------------------

        faithfulness_response = (
            self._invoke_with_fallback(
                self.faithfulness_chain,
                self.faithfulness_fallback_chain,
                {
                    "context":
                        context,

                    "answer":
                        answer
                },
                stage=
                    "faithfulness",

                second_fallback_chain=
                    self.faithfulness_second_fallback_chain
            )
        )


        try:

            faithfulness_data = (
                json.loads(
                    extract_text(faithfulness_response.content)
                )
            )


        except (
            json.JSONDecodeError,
            ValueError,
            TypeError
        ):

            faithfulness_data = {
                "supported_claims":
                    [],

                "unsupported_claims":
                    [],

                "reason":
                    "Faithfulness output "
                    "could not be parsed."
            }


        # ----------------------------------------------
        # Robust claim parsing
        # ----------------------------------------------

        def clean_claims(items):

            if not isinstance(
                items,
                list
            ):
                return []


            cleaned = []


            for item in items:

                if isinstance(
                    item,
                    str
                ):

                    cleaned.append(
                        item
                    )


                elif isinstance(
                    item,
                    dict
                ):

                    claim = item.get(
                        "claim"
                    )


                    if claim:

                        cleaned.append(
                            str(claim)
                        )


            return cleaned


        supported_claims = (
            clean_claims(
                faithfulness_data.get(
                    "supported_claims",
                    []
                )
            )
        )


        unsupported_claims = (
            clean_claims(
                faithfulness_data.get(
                    "unsupported_claims",
                    []
                )
            )
        )


        # ----------------------------------------------
        # Backward compatibility
        # ----------------------------------------------

        if (
            not supported_claims
            and not unsupported_claims
        ):

            old_claims = (
                faithfulness_data.get(
                    "claims",
                    []
                )
            )


            if isinstance(
                old_claims,
                list
            ):

                for item in old_claims:

                    if not isinstance(
                        item,
                        dict
                    ):
                        continue


                    claim = item.get(
                        "claim",
                        ""
                    )


                    supported = item.get(
                        "supported",
                        False
                    )


                    if not claim:
                        continue


                    if supported:

                        supported_claims.append(
                            claim
                        )


                    else:

                        unsupported_claims.append(
                            claim
                        )


        # ----------------------------------------------
        # Faithfulness score
        # ----------------------------------------------

        total_claims = (
            len(
                supported_claims
            )
            + len(
                unsupported_claims
            )
        )


        if total_claims > 0:

            faithfulness = (
                len(
                    supported_claims
                )
                / total_claims
            )


        else:

            faithfulness = 0.0


        return {
            "answer_relevance":
                relevance_score / 2,

            "faithfulness":
                faithfulness,

            "supported_claims":
                supported_claims,

            "unsupported_claims":
                unsupported_claims,

            "relevance_reason":
                relevance_data.get(
                    "reason",
                    ""
                ),

            "faithfulness_reason":
                faithfulness_data.get(
                    "reason",
                    ""
                )
        }


    # --------------------------------------------------
    # Optional diagnostics
    # --------------------------------------------------

    def get_fallback_stats(
        self
    ):

        return {
            "level_1_fallbacks":
                self.fallback_count,

            "level_2_fallbacks":
                self.second_fallback_count,

            "events":
                list(
                    self.fallback_events
                )
        }