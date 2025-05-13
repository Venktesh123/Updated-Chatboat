import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

class PrepareVectorDB:
    def __init__(self, file_path, chunk_size, chunk_overlap, embedding_model, vectordb_dir, collection_name):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name

    def validate_file(self) -> bool:
        if not os.path.exists(self.file_path):
            print(f"❌ Error: File '{self.file_path}' does not exist.")
            return False
        if not self.file_path.lower().endswith(('.txt', '.text')):
            print(f"❌ Error: File '{self.file_path}' is not a text file.")
            return False
        return True

    def run(self) -> None:
        if not self.validate_file():
            return

        if not os.path.exists(self.vectordb_dir):
            os.makedirs(self.vectordb_dir)
            print(f"📁 Directory '{self.vectordb_dir}' was created.")

            loader = TextLoader(self.file_path, encoding='utf-8')
            docs_list = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs_list)

            vectordb = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=GoogleGenerativeAIEmbeddings(model=self.embedding_model),
                persist_directory=self.vectordb_dir
            )
            print("✅ VectorDB created and saved.")
            print(f"🔢 Number of vectors: {vectordb._collection.count()}")
        else:
            print(f"ℹ️ Directory '{self.vectordb_dir}' already exists.")

if __name__ == "__main__":
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    prepare_db_instance = PrepareVectorDB(
        file_path="cleaned_transcript.txt",
        chunk_size=5000,
        chunk_overlap=1000,
        embedding_model='models/embedding-001',
        vectordb_dir='vectordb',
        collection_name='chroma'
    )
    prepare_db_instance.run()
    
