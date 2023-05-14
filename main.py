import os
import sys

from typing import Tuple
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain


class QnABot():
    def __init__(self):
        self.__vector_db = Chroma(
            persist_directory="db", embedding_function=OpenAIEmbeddings())

    def get_answer(self, query: str) -> Tuple[str, str]:
        chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(
            model_name="gpt-3.5-turbo", temperature=0.8, max_tokens=100,
            openai_api_key=os.environ['OPENAI_API_KEY']),
            chain_type="stuff",
            retriever=self.__vector_db.as_retriever())
        chain_result = chain(
            {"question": query}, return_only_outputs=True)
        return chain_result['answer'], chain_result['sources']


if __name__ == '__main__':
    question = sys.argv[1]
    print(f'QnA Bot Question: {question}')
    bot = QnABot()
    answer, source = bot.get_answer(question)
    print(f'Question: {question} Answer: {answer} Source: {source}')
