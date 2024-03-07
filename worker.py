import os

# Import necessary modules from langchain
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import OpenAI

# Load environment variables
load_dotenv()

# Initialize global variables
conversation_retrieval_chain = None
chat_history = []
llm = None
llm_embeddings = None

# Function to initialize the language model and its embeddings
def init_llm():
    global llm, llm_embeddings
    # Initialize the language model with the OpenAI API key
    api_key="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"
    llm = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=api_key)
    # Initialize the embeddings for the language model
    llm_embeddings = OpenAIEmbeddings(openai_api_key = api_key)

# Function to process a PDF document
def process_document(document_path):
    global conversation_retrieval_chain, llm, llm_embeddings
    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # Create a vector store from the document chunks
    db = Chroma.from_documents(texts, llm_embeddings)
    # Create a retriever interface from the vector store
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # Create a conversational retrieval chain from the language model and the retriever
    conversation_retrieval_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history
    # Pass the prompt and the chat history to the conversation_retrieval_chain object
    result = conversation_retrieval_chain({"question": prompt, "chat_history": chat_history})
    chat_history.append((prompt, result["answer"]))
    # Return the model's response
    return result['answer']

# Initialize the language model
init_llm()