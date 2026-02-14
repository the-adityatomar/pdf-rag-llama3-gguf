from llama_index.core import SimpleDirectoryReader


def load_pdf(pdf_path):
    """
    Loads a PDF file and returns LlamaIndex Document objects.
    """

    documents = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()

    return documents
