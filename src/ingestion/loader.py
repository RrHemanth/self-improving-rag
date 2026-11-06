from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
}


def build_document_id(path: Path) -> str:
    """
    Create a simple document identifier from the filename.
    """

    return path.stem.lower().replace(
        " ",
        "_"
    )


def validate_file(path: Path):
    """
    Check that the file exists and uses a supported extension.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, got: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            f"Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

# PDF

def load_pdf(file_path: str) -> list[dict]:
    """
    Load a PDF page-by-page.
    """

    path = Path(file_path)

    validate_file(
        path
    )

    reader = PdfReader(
        path
    )

    document_id = build_document_id(
        path
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append({
            "text":
                text,

            "source":
                path.name,

            "document_id":
                document_id,

            "page":
                page_number
        })

    return documents

# Plain text

def load_txt(file_path: str) -> list[dict]:
    """
    Load a plain-text file.
    """

    path = Path(file_path)

    validate_file(
        path
    )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [{
        "text":
            text,

        "source":
            path.name,

        "document_id":
            build_document_id(
                path
            ),

        "page":
            None
    }]

# Markdown

def load_markdown(file_path: str) -> list[dict]:
    """
    Load a Markdown file as text.
    """

    path = Path(file_path)

    validate_file(
        path
    )

    text = path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        return []

    return [{
        "text":
            text,

        "source":
            path.name,

        "document_id":
            build_document_id(
                path
            ),

        "page":
            None
    }]

# DOCX

def load_docx(file_path: str) -> list[dict]:
    """
    Load text from a DOCX file.

    Paragraphs and table cells are extracted.
    """

    try:
        from docx import Document

    except ImportError:
        raise ImportError(
            "DOCX support requires python-docx. "
            "Install it with: pip install python-docx"
        )

    path = Path(file_path)

    validate_file(
        path
    )

    document = Document(
        path
    )

    text_parts = []


    # Paragraphs
    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            text_parts.append(
                text
            )


    # Tables
    for table in document.tables:

        for row in table.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:
                text_parts.append(
                    " | ".join(
                        cells
                    )
                )


    text = "\n\n".join(
        text_parts
    ).strip()


    if not text:
        return []


    return [{
        "text":
            text,

        "source":
            path.name,

        "document_id":
            build_document_id(
                path
            ),

        "page":
            None
    }]

# Single-document loader

def load_document(
    file_path: str
) -> list[dict]:
    """
    Load one supported document.

    This function preserves compatibility with
    the existing RAG pipeline.
    """

    path = Path(
        file_path
    )

    extension = (
        path.suffix.lower()
    )


    if extension == ".pdf":
        return load_pdf(
            file_path
        )


    if extension == ".txt":
        return load_txt(
            file_path
        )


    if extension == ".md":
        return load_markdown(
            file_path
        )


    if extension == ".docx":
        return load_docx(
            file_path
        )


    raise ValueError(
        f"Unsupported file type: {extension}. "
        f"Supported types: "
        f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )

# Multiple-document loader

def load_documents(
    file_paths: list[str]
) -> list[dict]:
    """
    Load multiple documents and combine them
    into one normalized document collection.
    """

    documents = []


    for file_path in file_paths:

        loaded = load_document(
            file_path
        )

        documents.extend(
            loaded
        )


    return documents

# Directory loader

def load_directory(
    directory_path: str
) -> list[dict]:
    """
    Load all supported documents from a directory.
    """

    directory = Path(
        directory_path
    )


    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: "
            f"{directory}"
        )


    if not directory.is_dir():
        raise ValueError(
            f"Expected a directory, got: "
            f"{directory}"
        )


    file_paths = [
        str(path)
        for path in sorted(
            directory.iterdir()
        )
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]


    return load_documents(
        file_paths
    )