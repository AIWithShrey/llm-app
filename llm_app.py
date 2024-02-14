import streamlit as st
import os
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import tempfile

# Initialize Llama model
model_path = "models/openhermes-7b-v2.5/DESIRED_QUANTISED_MODEL_HERE"
model = LlamaCpp(model_path=model_path, 
                 chat_format="llama-2", 
                 n_gpu_layers=-1, 
                 n_batch=2048, 
                 n_ctx=2048,
                 f16_kv=True)

template = """Answer the questions based on the context below. If the
questions cannot be answered using the information provided answer
with "I don't know" or "I cannot answer that".

Context: You are an assistant that answers questions accurately and concisely. 
You do not help with any requests that are inappropriate or illegal. You are friendly.

Question: {query}

Answer: """

prompt_template = PromptTemplate(input_variables=['query'], template=template)

# Streamlit app setup
st.set_page_config(layout="wide")
st.title("LLMBot")
st.text("Chatbot based on Mistral-7B")
col1, col2 = st.columns([0.9, 0.6])
with col1:
    user_input = st.text_input("You:", "")
with col2:
    uploaded_file = st.file_uploader("Choose a document", type=['pdf', 'docx', 'txt'])

# Function to get a response from the model
def get_response(message):
    response = model.predict(prompt_template.format(query=message))
    return response#['choices'][0]['message']['content']

# Function to load and chunk document data
def load_document(file):
    # Create a temporary file to save the uploaded file's content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
        tmp.write(file.getvalue())  # Write the content of the uploaded file to the temporary file
        tmp_path = tmp.name  # Store the path of the temporary file

    # Now use tmp_path with the appropriate loader
    if tmp_path.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    elif tmp_path.endswith('.docx'):
        loader = Docx2txtLoader(tmp_path)
    elif tmp_path.endswith('.txt'):
        loader = TextLoader(tmp_path)
    else:
        st.error('Document format is not supported!')
        return None

    data = loader.load()
    os.unlink(tmp_path)  # Clean up the temporary file
    return data

def chunk_data(data, chunk_size=512):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=2)
    chunks = text_splitter.split_documents(data)
    return chunks

# Function to insert or fetch embeddings
def insert_or_fetch_embeddings(chunks):
    embeddings_model = GPT4AllEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    return vector_store

# Function to get response with document
def get_response_with_doc(user_input, uploaded_file):
    data = load_document(uploaded_file)
    if data:
        chunks = chunk_data(data)
        #index_name = "llm-app"  # Define a consistent index name
        vector_store = insert_or_fetch_embeddings(chunks)
        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
        chain = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)
        answer = chain.invoke(user_input)
        return answer['result']
    return "Could not process document."

# Display bot response based on user input and/or uploaded file
if uploaded_file is not None and user_input:
    bot_response = get_response_with_doc(user_input, uploaded_file)
else:
    bot_response = get_response(user_input) if user_input else ""

st.text_area("Joe:", value=bot_response, height=200, key="bot_response_area")
