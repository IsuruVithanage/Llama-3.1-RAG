from langchain_community.document_loaders import PyPDFLoader  # For loading PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # Free embeddings from Hugging Face
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Path to the PDF file
pdf_path = "diet.pdf"

# Load the PDF content
pdf_loader = PyPDFLoader(pdf_path)
docs = pdf_loader.load()

# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs)

# Create embeddings for documents and store them in a vector store using HuggingFaceEmbeddings
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=HuggingFaceEmbeddings(),  # Free Hugging Face embeddings
)
retriever = vectorstore.as_retriever(k=4)

# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()


# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)

# Example usage
question = "I am 24 years old and my weight is 57, my LDL level is high what are the Sri Lankan foods and other things that I shoild do inorder to lower the LDL lecel?"
answer = rag_application.run(question)
print("Question:", question)
print("Answer:", answer)
