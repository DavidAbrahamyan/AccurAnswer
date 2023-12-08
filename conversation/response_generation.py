from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from typing import Any, List, Tuple
import cohere

import os
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
PINECONE_512_INDEX_API_KEY = os.getenv('PINECONE_512_INDEX_API_KEY')
PINECONE_1024_INDEX_API_KEY = os.getenv('PINECONE_1024_INDEX_API_KEY')
PINECONE_ENVIRONMENT_512 = os.getenv('PINECONE_ENVIRONMENT_512')
PINECONE_ENVIRONMENT_1024 = os.getenv('PINECONE_ENVIRONMENT_1024')
PINECONE_512_INDEX_NAME = os.getenv('PINECONE_512_INDEX_NAME')
PINECONE_1024_INDEX_NAME = os.getenv('PINECONE_1024_INDEX_NAME')

co = cohere.Client(COHERE_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def parse_rerank_to_list(k, query, documents):
    rerank_response = co.rerank(
        model='rerank-english-v2.0',
        query=query,
        documents=documents,
        top_n=k,
    )
    rerank_list = []

    for item in rerank_response:
        rerank_list.append(item.document["text"])
    return rerank_list


def parse_doc_to_list(doc):
    input_string = str(doc)

    document_strings = input_string.split("), Document(")

    document_strings[0] = document_strings[0].replace("Document(", "")
    document_strings[-1] = document_strings[-1].replace("')", "")

    page_contents = [doc.split("page_content=")[1] for doc in document_strings]

    return page_contents


def get_response_rerank_512(prompt: str, chat_history: List[Tuple[str, str]]) -> Any:
    """

    :param prompt: The user-given question (prompt)
    :param chat_history: The previous history of the conversation between AccurAnswer and user
    :return: Answer to the user-given question
    """

    pinecone.init(environment=PINECONE_ENVIRONMENT_512, api_key=PINECONE_512_INDEX_API_KEY)

    search_helper = Pinecone.from_existing_index(index_name=PINECONE_512_INDEX_NAME, embedding=embeddings)

    # Modify the value of k before to retrieve more (or less) chunks from the database that will be re-ranked in the next step.
    pinecone_retriever = search_helper.as_retriever(search_kwargs={"k": 27})

    # Modify the value of top_n below to the number of chunks that you want to use for generating response
    compressor_4 = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=4)

    openai_compression_retriever_4 = ContextualCompressionRetriever(
        base_compressor=compressor_4, base_retriever=pinecone_retriever
    )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=OPENAI_API_KEY, verbose=True, temperature=0)

    chain4 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_4)

    result = chain4({"question": prompt, "chat_history": chat_history})

    return result


def get_response_rerank_1024(prompt: str, chat_history: List[Tuple[str, str]]) -> Any:
    """

    :param prompt: The user-given question (prompt)
    :param chat_history: The previous history of the conversation between AccurAnswer and user
    :return: Answer to the user-given question
    """

    pinecone.init(environment=PINECONE_ENVIRONMENT_1024, api_key=PINECONE_1024_INDEX_API_KEY)

    search_helper = Pinecone.from_existing_index(index_name=PINECONE_1024_INDEX_NAME, embedding=embeddings)

    # Modify the value of k before to retrieve more (or less) chunks from the database that will be re-ranked in the next step.
    pinecone_retriever = search_helper.as_retriever(search_kwargs={"k": 27})

    # Modify the value of top_n below to the number of chunks that you want to use for generating response
    compressor_4 = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=4)

    openai_compression_retriever_4 = ContextualCompressionRetriever(
        base_compressor=compressor_4, base_retriever=pinecone_retriever
    )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=OPENAI_API_KEY, verbose=True, temperature=0)

    chain4 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_4)

    result = chain4({"question": prompt, "chat_history": chat_history})

    return result
