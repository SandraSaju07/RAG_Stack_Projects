from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import streamlit as st

template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
""" 

pdfs_directory = "./PDFs"

embeddings = OllamaEmbeddings(model="deepseek-r1:latest")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:latest")


def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file):
    loader = PDFPlumberLoader(file)
    documents = loader.load()
    return documents


def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_docs(documents):
    vector_store.add_documents(documents)
    

def retrieve_docs(query):
    return vector_store.similarity_search(query)


def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    response = chain.invoke({"question": question, "context": context})
    return response


# Streamlit UI
uploaded_file = st.file_uploader(
    "Upload PDF", 
    type=["pdf"], 
    accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_docs = split_text(documents)
    index_docs(chunked_docs)
    
    question = st.chat_input()
    
    if question:
        st.chat_message("User").write(question)
        related_docs = retrieve_docs(question)
        answer = answer_question(question, related_docs)
        st.chat_message("Assistant").write(answer)
