import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.app.pipeline import build_agent


st.set_page_config(
    page_title="Self-Improving RAG",
    page_icon="🧠",
    layout="wide",
)

st.title("Self-Improving RAG")
st.caption(
    "Upload documents, ask questions, and let the system adapt "
    "its retrieval strategy from experience."
)


st.sidebar.header("Model settings")

provider_label = st.sidebar.selectbox(
    "LLM provider",
    ["Ollama (local)", "OpenAI", "Anthropic (Claude)", "Google Gemini"],
)

provider = {
    "Ollama (local)": "ollama",
    "OpenAI": "openai",
    "Anthropic (Claude)": "anthropic",
    "Google Gemini": "gemini",
}[provider_label]

model = None
api_key = None

if provider == "ollama":
    st.sidebar.caption(
        "Uses local Qwen 3B for generation "
        "and Qwen 7B for evaluation."
    )
else:
    default_model = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5",
        "gemini": "gemini-3.1-flash-lite",
    }[provider]

    model = st.sidebar.text_input(
        "Model",
        value=default_model,
    )

    api_key = st.sidebar.text_input(
        "API key",
        type="password",
        placeholder="Paste your API key",
    )

    st.sidebar.caption(
        "Your key is held in the active server session, "
        "not saved to project files."
    )

    st.sidebar.warning(
        "Cloud API calls may incur charges. "
        "Relevant document excerpts are sent "
        "to your selected provider."
    )

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_config" not in st.session_state:
    st.session_state.active_config = None

key_fingerprint = (
    hashlib.sha256(api_key.encode()).hexdigest()
    if api_key else None
)

current_config = (
    provider,
    model,
    key_fingerprint,
)

if (
    st.session_state.pipeline is not None
    and st.session_state.active_config != current_config
):
    st.session_state.pipeline = None
    st.session_state.messages = []
    st.session_state.active_config = None

    st.info(
        "Model settings changed. "
        "Process your documents again."
    )


uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "md", "docx"],
    accept_multiple_files=True,
)


if st.button(
    "Process documents",
    disabled=(
        not uploaded_files
        or (
            provider != "ollama"
            and (
                not model.strip()
                or not api_key.strip()
            )
        )
    ),
):
    try:
        with st.spinner("Processing documents..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []

                for uploaded_file in uploaded_files:
                    path = Path(temp_dir) / uploaded_file.name

                    path.write_bytes(
                        uploaded_file.getvalue()
                    )

                    file_paths.append(
                        str(path)
                    )

                pipeline = build_agent(
                    file_paths,
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )

        st.session_state.pipeline = pipeline
        st.session_state.messages = []
        st.session_state.active_config = current_config

        st.success(
            f"Processed {len(uploaded_files)} document(s) "
            f"into {len(pipeline['chunks'])} chunks."
        )

    except Exception as error:
        st.error(
            f"Could not process documents: {error}"
        )


pipeline = st.session_state.pipeline


if pipeline is not None:
    with st.expander("Document collection"):
        st.write(
            f"Corpus ID: `{pipeline['corpus_id']}`"
        )
        st.write(
            f"Document units: {len(pipeline['documents'])}"
        )
        st.write(
            f"Chunks: {len(pipeline['chunks'])}"
        )
        st.write(
            f"Learned experiences: "
            f"{len(pipeline['agent'].memory.experiences)}"
        )


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

        if message.get("diagnostics"):
            diagnostics = message[
                "diagnostics"
            ]

            with st.expander(
                "Retrieval diagnostics"
            ):
                st.write(
                    f"Mode: `{diagnostics['mode']}`"
                )

                st.write(
                    "Initial strategy: "
                    f"`{diagnostics['initial_strategy']}`"
                )

                st.write(
                    "Best strategy: "
                    f"`{diagnostics['best_strategy']}`"
                )

                st.write(
                    f"Attempts: "
                    f"{len(diagnostics['attempts'])}"
                )

                attempt_rows = []

                for attempt in diagnostics[
                    "attempts"
                ]:
                    attempt_rows.append(
                        {
                            "Strategy": attempt[
                                "strategy"
                            ],
                            "Retrieval": round(
                                attempt[
                                    "retrieval_relevance"
                                ],
                                3,
                            ),
                            "Relevance": round(
                                attempt[
                                    "answer_relevance"
                                ],
                                3,
                            ),
                            "Faithfulness": round(
                                attempt[
                                    "faithfulness"
                                ],
                                3,
                            ),
                            "Score": round(
                                attempt[
                                    "overall_score"
                                ],
                                3,
                            ),
                            "Success": attempt[
                                "success"
                            ],
                        }
                    )

                st.dataframe(
                    attempt_rows,
                    use_container_width=True,
                    hide_index=True,
                )


if pipeline is None:
    st.info(
        "Upload one or more documents and process them "
        "before asking a question."
    )

else:
    query = st.chat_input(
        "Ask a question about your documents"
    )

    if query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
            }
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner(
                "Searching, evaluating, and generating..."
            ):
                try:
                    result = pipeline[
                        "agent"
                    ].run(query)

                    cited_answer = pipeline[
                        "generator"
                    ].add_citations(
                        result["answer"],
                        result["results"],
                    )

                    st.markdown(
                        cited_answer
                    )

                    with st.expander(
                        "Retrieval diagnostics"
                    ):
                        st.write(
                            f"Mode: `{result['mode']}`"
                        )

                        st.write(
                            "Initial strategy: "
                            f"`{result['initial_strategy']}`"
                        )

                        st.write(
                            "Best strategy: "
                            f"`{result['best_strategy']}`"
                        )

                        st.write(
                            f"Attempts: "
                            f"{len(result['attempts'])}"
                        )

                        attempt_rows = []

                        for attempt in result[
                            "attempts"
                        ]:
                            attempt_rows.append(
                                {
                                    "Strategy": attempt[
                                        "strategy"
                                    ],
                                    "Retrieval": round(
                                        attempt[
                                            "retrieval_relevance"
                                        ],
                                        3,
                                    ),
                                    "Relevance": round(
                                        attempt[
                                            "answer_relevance"
                                        ],
                                        3,
                                    ),
                                    "Faithfulness": round(
                                        attempt[
                                            "faithfulness"
                                        ],
                                        3,
                                    ),
                                    "Score": round(
                                        attempt[
                                            "overall_score"
                                        ],
                                        3,
                                    ),
                                    "Success": attempt[
                                        "success"
                                    ],
                                }
                            )

                        st.dataframe(
                            attempt_rows,
                            use_container_width=True,
                            hide_index=True,
                        )

                except Exception as error:
                    st.error(
                        f"Could not answer the question: {error}"
                    )
                    st.stop()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": cited_answer,
                "diagnostics": {
                    "mode": result["mode"],
                    "initial_strategy": result[
                        "initial_strategy"
                    ],
                    "best_strategy": result[
                        "best_strategy"
                    ],
                    "attempts": result[
                        "attempts"
                    ],
                },
            }
        )