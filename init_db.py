import os
import sys

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


def index_directory_to_vector_db(directory: str):
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            print(f'Indexing file {file}')
            index_file_to_vector_db(os.path.join(directory, file), "db")


if __name__ == '__main__':
    print('Initialize the vector DB!')
    index_directory_to_vector_db(sys.argv[1])
