from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


def index_file_to_vector_db(file_path: str):
    """
    This script is used to initialize the vector DB. It takes a text file as input and splits it
    into chunks of 1000 characters.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        input_file = file.read()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(input_file)

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[
        {"source": f"Text chunk {i} of {len(texts)}"} for i in range(len(texts))],
        persist_directory="db")
    docsearch.persist()


if __name__ == '__main__':
    print('Initialize the vector DB!')
    index_file_to_vector_db('state_of_the_union.txt')
