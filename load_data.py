import pandas as pd

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

import os
from dotenv import load_dotenv, find_dotenv
import logging
load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_512_INDEX_API_KEY = os.getenv('PINECONE_512_INDEX_API_KEY')
PINECONE_1024_INDEX_API_KEY = os.getenv('PINECONE_1024_INDEX_API_KEY')
PINECONE_512_INDEX_NAME = os.getenv('PINECONE_512_INDEX_NAME')
PINECONE_1024_INDEX_NAME = os.getenv('PINECONE_1024_INDEX_NAME')
PINECONE_ENVIRONMENT_512 = os.getenv('PINECONE_ENVIRONMENT_512')
PINECONE_ENVIRONMENT_1024=os.getenv('PINECONE_ENVIRONMENT_1024')

pd.options.display.max_colwidth = None

logging.basicConfig(
    level="DEBUG",
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_data_512(text_data, document_data):
    """
    A function to save hugging face transformer models' data and LangChain Documentation into Pinecone using a chunk size of 512
    :param text_data:
    :param document_data:
    :return: None
    """
    # os.environ["HUGGINGFACE_API_TOKEN"] = "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"

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, disallowed_special=())
    pinecone.init(
        api_key=PINECONE_512_INDEX_API_KEY,
        environment=PINECONE_ENVIRONMENT_512
    )
    index_name = PINECONE_512_INDEX_NAME
    logging.info("Started loading hugging face transformers data into Pinecone with OpenAI embeddings model")
    Pinecone.from_texts(text_data, embeddings, index_name=index_name)
    logging.info("Successfully loaded hugging face transformers data into Pinecone with OpenAI embeddings model")

    logging.info("Started loading document data (Langchain) into Pinecone with OpenAI embeddings model")
    Pinecone.from_documents(document_data, embeddings, index_name=index_name)
    logging.info("Successfully loaded document data into Pinecone with OpenAI embeddings model")


def load_data_1024(text_data, document_data):
    """
    A function to save hugging face transformer models' data and LangChain Documentation into Pinecone using a chunk size of 1024
    :param text_data:
    :param document_data:
    :return: None
    """

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, disallowed_special=())
    pinecone.init(
        api_key=PINECONE_1024_INDEX_API_KEY,
        environment=PINECONE_ENVIRONMENT_1024
    )
    index_name = PINECONE_1024_INDEX_NAME
    logging.info("Started loading hugging face transformers data into Pinecone with OpenAI embeddings model")
    Pinecone.from_texts(text_data, embeddings, index_name=index_name)
    logging.info("Successfully loaded hugging face transformers data into Pinecone with OpenAI embeddings model")

    logging.info("Started loading document data (Langchain) into Pinecone with OpenAI embeddings model")
    Pinecone.from_documents(document_data, embeddings, index_name=index_name)
    logging.info("Successfully loaded document data into Pinecone with OpenAI embeddings model")


def split_data_512(text_data_path: str, document_data_path: str):
    """
    A function to divide hugging face transformer models' data and LangChain Documentation into chunks a chunk size of 1024
    :param text_data_path: path to the hugging face transformer models' data
    :param document_data_path: path to the LangChain documentation data
    :return: A tuple of HuggingFace data chunks and LangChain data chunks
    """
    loader = ReadTheDocsLoader(path=document_data_path)
    raw_documents = loader.load()

    logging.info(f"Loaded {len(raw_documents)} documents from the LangChain Documentation.") # 2380 documents.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=40, separators=["\n\n", "\n", "", " "]
    )

    hugging_face_transformers_path = text_data_path
    hugging_face_transformers_data = ""
    if os.path.exists(hugging_face_transformers_path):
        with open(hugging_face_transformers_path, "r") as hf_transformers_file:
            hugging_face_transformers_data = hf_transformers_file.read()
    lang_chain_chunks = text_splitter.split_documents(raw_documents)
    logging.info(f"Langchain documentation Split into {len(lang_chain_chunks)} chunks using a chunk size of 512") # 30235 chunks

    hugging_face_transformers_chunks = text_splitter.split_text(hugging_face_transformers_data)
    logging.info(f"The HuggingFace Transformers File was split into {len(hugging_face_transformers_chunks)} chunks using a chunk size of 512.") #60901 chunks.

    return lang_chain_chunks, hugging_face_transformers_chunks


def split_data_1024(text_data_path, document_data_path):
    """
    A function to divide hugging face transformer models' data and LangChain Documentation into chunks a chunk size of 1024
    :param text_data_path:
    :param document_data_path:
    :return: A tuple of HuggingFace data chunks and LangChain data chunks
    """
    loader = ReadTheDocsLoader(path=document_data_path)
    raw_documents = loader.load()
    logging.info(f"Loaded {len(raw_documents)} documents from the LangChain Documentation.")  # 2380 documents.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=30, separators=["\n\n", "\n", "", " "]
    )

    lang_chain_chunks = text_splitter.split_documents(raw_documents)
    logging.info(f"Langchain documentation Split into {len(lang_chain_chunks)} chunks with chunk size of 1024")  # 14834 chunks

    hugging_face_transformers_path = text_data_path
    hugging_face_transformers_text = ""
    if os.path.exists(hugging_face_transformers_path):
        with open(hugging_face_transformers_path, "r") as hf_transformers_file:
            hugging_face_transformers_text = hf_transformers_file.read()
    hugging_face_transformers_chunks = text_splitter.split_text(hugging_face_transformers_text)
    logging.info(f"The HuggingFace Transformers File was split into {len(hugging_face_transformers_chunks)} chunks with chunk size of 1024.")  # 29714 chunks.

    return lang_chain_chunks, hugging_face_transformers_chunks


if __name__ == '__main__':
    lang_chain_chunks_512, hugging_face_transformers_chunks_512 = split_data_512("hugging_face_transformers.txt", "langchain-docs")
    lang_chain_chunks_1024, hugging_face_transformers_chunks_1024 = split_data_1024("", "")
    load_data_512(hugging_face_transformers_chunks_512, lang_chain_chunks_512)
    load_data_1024(hugging_face_transformers_chunks_1024, lang_chain_chunks_1024)
