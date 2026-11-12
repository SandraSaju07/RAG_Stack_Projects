import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-miniLM-L6-v2", 
    model_kwargs={"device": "cpu"}
    )
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_tokens=512, max_retries=3)

PERSIST_DIRECTORY = "./qdrant_star_wars"
COLLECTION_NAME = "star_wars_scripts"

def fetch_start_wars_data(url, movie_title):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        data = soup.find('pre').get_text()
        return Document(page_content=data, metadata={'title': movie_title})

def main():
    qdrant_client = QdrantClient(path=PERSIST_DIRECTORY)

    try:
        qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        vector_store = QdrantVectorStore(
            collection_name=COLLECTION_NAME,
            embedding=embeddings,
            client=qdrant_client,
        )

    except Exception as e:
        qdrant_client.close()
        star_wars_scripts = [
            {
                "title": "Star Wars: A New Hope",
                "url": "https://www.imsdb.com/scripts/Star-Wars-A-New-Hope.html",
            },
            {
                "title": "Star Wars: The Empire Strikes Back",
                "url": "https://www.imsdb.com/scripts/Star-Wars-The-Empire-Strikes-Back.html",
            },
            {
                "title": "Star Wars: Return of the Jedi",
                "url": "https://www.imsdb.com/scripts/Star-Wars-Return-of-the-Jedi.html",
            },
        ]

        script_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250,
            add_start_index=True,
            separators=["\nINT", "\nEXT", "\n\n", "\n", " ", ""],
        )

        all_chunks = []

        for script in star_wars_scripts:
            document = fetch_start_wars_data(script["url"], script["title"])
            chunks = script_splitter.split_documents([document])
            all_chunks.extend(chunks)
            print(f"Processed {script['title']} into {len(chunks)} chunks.")

        vector_store = QdrantVectorStore.from_documents(
            all_chunks,
            embedding=embeddings,
            path=PERSIST_DIRECTORY,
            collection_name=COLLECTION_NAME,
            force_recreate=True
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 8})

    template = """
    You are a Star Wars Movie Script Expert. You only the following scripts to answer the questions.
    If the answer is partly contained provide the best possible answer based on the context.
    If the answer is not in the movie scripts, say I don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Welcome to the Star Wars RAG System! Ask any question about the Star Wars movies.")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break

        response = rag_chain.invoke(query)
        print(f"\nStar Wars RAG System: {response}")

if __name__ == "__main__":
    main()