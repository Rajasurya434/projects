

!pip install langchain langchain-community sentence-transformers faiss-cpu transformers pypdf

import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain.schema import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DocumentQA:
    def __init__(self, pdf_path, huggingface_api_token):
        """Initialize with Hugging Face API and process the PDF."""
        self.pdf_path = pdf_path
        self.huggingface_api_token = huggingface_api_token


        logging.info("Initializing sentence embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )


        logging.info("Connecting to Hugging Face LLM API...")
        self.llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            top_k=50,
            top_p=0.95,
            huggingfacehub_api_token=self.huggingface_api_token,
        )


        self.qa_prompt_template = """
        <s>[INST] Use the following context to answer the question. If unsure, say "I don't know."

        Context:
        {context}

        Question: {question} [/INST]
        """
        self.prompt = PromptTemplate(template=self.qa_prompt_template, input_variables=["context", "question"])


        self.vectorstore = None


        self.process_pdf()

    def process_pdf(self):
        """Load, split, and store PDF in FAISS vector database."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file '{self.pdf_path}' not found!")

        logging.info(f"Loading PDF: {self.pdf_path}")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)


        logging.info("Creating FAISS vector store...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        logging.info("Vector store created successfully!")

    def setup_qa_chain(self):
        """Set up the QA retrieval system."""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Process PDF first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        logging.info("QA Chain setup complete!")

    def ask_question(self, question):
        """Ask a question and get an answer."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first.")

        response = self.qa_chain({"query": question})
        return {
            "answer": response["result"],
            "source_documents": response["source_documents"]
        }

pdf_file = "/content/short-fiction-story.pdf"


huggingface_api_token = "give ur token"


qa_system = DocumentQA(pdf_file, huggingface_api_token)


qa_system.setup_qa_chain()


while True:
    user_question = input("\nAsk a question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        print("Exiting...")
        break

    response = qa_system.ask_question(user_question)
    print("\n🔹 Answer:", response["answer"])
    print("\n📖 Source Documents:", [doc.metadata for doc in response["source_documents"]])