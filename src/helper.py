from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings



#extract text from pdf files
def load_pdf_files(data):
    loader = DirectoryLoader(data, 
                             glob="*.pdf", 
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    that only contain source in metadata and the original page content.
    """

    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")

        minimal_doc = Document(
            page_content=doc.page_content,
            metadata={"source": src}
        )

        minimal_docs.append(minimal_doc)

    return minimal_docs



#split documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=20)
    text_chunks = text_splitter.split_documents(minimal_docs)
    return text_chunks



def download_embeddings():
    """
    Download and return the HuggingFace embedding model.
    """
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name
    )
    
    return embeddings

