from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_embeddings
from pinecone import Pinecone,PodSpec
from pinecone import ServerlessSpec
from langchain_painecone import PineconeVectorStore


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data  = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
text_chunks = text_splitter.split_documents(minimal_docs)

embedding = download_embeddings()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=Pinecone_api_key)

index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


index = pc.Index(index_name)


docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)



#alreday created the pincone index, so no need this code again 
#you can delete the pincone index from pinecone dashboard and run this code to create the index again if you want to test the code again.