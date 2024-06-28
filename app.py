import streamlit as st
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
import os
from dotted_dict import DottedDict
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# Streamlit page configuration
st.set_page_config(
    page_title="Q&A",
    page_icon=":speech_balloon:",
    layout="centered",
    initial_sidebar_state="auto"
)

# CSS for custom styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Azure Configuration
azure_config = {
    "model_deployment": "GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO",
    "embedding_deployment": "ADA_RAG_DONO_DEMO",
    "embedding_name": "text-embedding-ada-002",
    "api_key":"f6f4b8aec16b4094bfe0b8e063dbf1a3",
    "api_version":"2024-02-01",
    "endpoint":"https://dono-rag-demo-resource-instance.openai.azure.com/",
    "model":"GPT_35_TURBO_DEMO_RAG_DEPLOYMENT_DONO"
}

# Function to initialize models
def models(azure_config):
    models = DottedDict()
    llm = AzureChatOpenAI(
        temperature=0,
        api_key=azure_config["api_key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["endpoint"],
        model=azure_config["model_deployment"],
        validate_base_url=False
    )
    embedding_model = AzureOpenAIEmbeddings(
        api_key=azure_config["api_key"],
        openai_api_version=azure_config["api_version"],
        azure_endpoint=azure_config["endpoint"],
        model=azure_config["embedding_deployment"]
    )
    models.llm = llm
    models.embedding_model = embedding_model
    return models

models = models(azure_config)

# Function to load Wikipedia data
def load_wikipedia(query):
    loader = WebBaseLoader(web_path=query)
    data = loader.load()
    return data

doc = load_wikipedia('https://en.wikipedia.org/wiki/FIFA_World_Cup')

# Function to chunk data
def chunking(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=30,
        length_function=len,
        is_separator_regex=False,
        separators=["."]
    )
    chunk = text_splitter.split_documents(data)
    return chunk

chunked_data = chunking(doc)

# Function to get vector store
def get_vector_store(text_chunks, models):
    embeddings = models.embedding_model
    vectordb = FAISS.from_documents(chunked_data, embeddings)
    return vectordb

# Function to get conversational chain
def get_conversational_chain(llm):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# Function to get answer to a query
def get_answer(query):
    document_search = get_vector_store(chunked_data, models)
    similar_docs = document_search.similarity_search(query, k=1)  # get closest chunks
    chain = get_conversational_chain(models.llm)
    answer = chain.invoke(input={"input_documents": similar_docs, "question": query}, return_only_outputs=True)
    return answer

# Streamlit UI
st.title("Question & Answer System")
st.header("Ask any question about the FIFA World Cup, and get detailed answers from Wikipedia!")

question = st.text_input("Ask your question here:", "")
submit = st.button("Submit")

if submit and question:
    with st.spinner('Searching for the answer...'):
        answer = get_answer(question)
    st.write("### Answer:")
    st.write(answer)
    
    # Adding chat history
    from langchain.memory import ChatMessageHistory
    history = ChatMessageHistory()
    history.add_user_message([question])
    history.add_ai_message([answer])
else:
    st.write("Please enter a question to get started.")
