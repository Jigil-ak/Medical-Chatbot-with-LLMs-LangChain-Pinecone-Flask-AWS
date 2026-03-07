from flask import Flask, render_template, request, jsonify
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from src.helper import download_embeddings
from src.prompt import system_prompt

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
import os

# load environment variables (Pinecone API key)
load_dotenv()

app = Flask(__name__)

# -------------------------
# Load embeddings
# -------------------------
embedding = download_embeddings()

index_name = "medical-chatbot"

# -------------------------
# Load Pinecone vector DB
# -------------------------
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)

# -------------------------
# Load FLAN-T5 model
# -------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

llm = HuggingFacePipeline(pipeline=pipe)

# -------------------------
# Prompt template
# -------------------------
prompt = PromptTemplate(
    template=system_prompt,
    input_variables=["context", "question"]
)

# -------------------------
# Retrieval QA chain
# -------------------------
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data["message"]

    result = qa.invoke({"query": query})
    answer = result["result"]

    return jsonify({"response": answer})


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)