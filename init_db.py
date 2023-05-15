import os
import sys

from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def index_file_to_vector_db(file_path: str, db_path: str):
    """
    This script is used to initialize the vector DB. It takes a text file as input and splits it
    into chunks of 1000 characters.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        input_file = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(input_file)

    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[
        {"source": f"Text chunk {i} of {len(texts)} of {file_path}"} for i in range(len(texts))],
        persist_directory=db_path)
    docsearch.persist()


def index_directory(input_path: str, db_path: str):
    txt_loader = DirectoryLoader(input_path, glob="**/*.txt")
    pdf_loader = DirectoryLoader(input_path, glob="**/*.pdf")
    loaders = [txt_loader, pdf_loader]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    print(f"Total number of documents: {len(documents)}")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        documents, embeddings, persist_directory=db_path)
    vectorstore.persist()


if __name__ == '__main__':
    print('Initialize the vector DB!')
    index_directory(sys.argv[1], sys.argv[2])
