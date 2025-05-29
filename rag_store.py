# rag_store.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./rag_db", embedding_function=embedding)
retriever = vectordb.as_retriever()

def add_to_rag(text: str):
    vectordb.add_documents([Document(page_content=text)])

def query_rag(user_query: str) -> str:
    # Retrieve documents from vector store
    docs = retriever.get_relevant_documents(user_query)
    combined_text = "\n".join([doc.page_content for doc in docs])

    # Summarize using an LLM (you can also use Gemini here)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)  # or ChatGoogleGenerativeAI
    summary_prompt = f"""
    Given the following information retrieved from tools:
    
    {combined_text}
    
    Summarize the key information in 3-5 sentences.
    """

    response = llm.predict(summary_prompt)
    return response
