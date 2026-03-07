# Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

#How to run?
#STEPS:
Clone the repository

git clone https://github.com/Jigil-ak/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git

#STEP 01- Create a conda environment after opening the repository
conda create -n medibot python=3.10 -y
conda activate medibot

#STEP 02- install the requirements
pip install -r requirements.txt

#STEP 03- Set the Pinecone API Key

Create a `.env` file in the root directory of the project and add your Pinecone API key.

PINECONE_API_KEY=your_pinecone_api_key_here

You can get your API key from the Pinecone dashboard:
https://app.pinecone.io


#STEP 04- Prepare the Vector Database

Before running the chatbot, you need to create embeddings from the medical documents and store them in the Pinecone vector database.

Run the following script:

python store_index.py

This step will:
- Load the medical PDF documents
- Split the documents into smaller chunks
- Convert the text into embeddings
- Store the embeddings in Pinecone


#STEP 05- Run the Flask Application

After the vector database is created, start the chatbot server.

python app.py


#STEP 06- Open the Chatbot

Once the server starts, open the following URL in your browser:

http://localhost:8080

You will see the Medical Chatbot interface where you can ask questions.


#Project Architecture

User Question  
↓  
Flask API  
↓  
Pinecone Vector Database  
↓  
Relevant Medical Context  
↓  
FLAN-T5 Language Model  
↓  
Generated Answer  


#Technologies Used

- Python
- LangChain
- Pinecone Vector Database
- HuggingFace Transformers
- FLAN-T5 (google/flan-t5-base)
- Flask
- HTML / CSS / JavaScript


#Example Questions

- What is blood pressure?
- What causes hypertension?
- What are the symptoms of diabetes?
- How is heart disease treated?


#Future Improvements

- Improve retrieval accuracy
- Add conversation memory
- Display source citations from documents
- Improve prompt engineering
- Deploy the application on cloud platforms