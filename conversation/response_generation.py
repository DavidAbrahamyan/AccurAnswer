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

    # Split the input string into individual Document strings
    document_strings = input_string.split("), Document(")

    # Remove any extra text around the Document strings
    document_strings[0] = document_strings[0].replace("Document(", "")
    document_strings[-1] = document_strings[-1].replace("')", "")

    # Extract the page_content values and create a list
    page_contents = [doc.split("page_content=")[1] for doc in document_strings]

    return page_contents


def get_response_rerank_512(prompt: str, chat_history: List[Tuple[str, str]]) -> Any:
    pinecone.init(environment=PINECONE_ENVIRONMENT_512, api_key=PINECONE_512_INDEX_API_KEY)

    search_helper = Pinecone.from_existing_index(index_name="doc-api-hf-data-index-512", embedding=embeddings)

    pinecone_retriever = search_helper.as_retriever(search_kwargs={"k": 27})
    # compressor_1 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=1)
    # compressor_2 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=2)
    compressor_4 = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=4)
    # compressor_8 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=8)
    # openai_compression_retriever_1 = ContextualCompressionRetriever(
    #     base_compressor=compressor_1, base_retriever=pinecone_retriever
    # )
    # openai_compression_retriever_2 = ContextualCompressionRetriever(
    #     base_compressor=compressor_2, base_retriever=pinecone_retriever
    # )
    openai_compression_retriever_4 = ContextualCompressionRetriever(
        base_compressor=compressor_4, base_retriever=pinecone_retriever
    )
    # openai_compression_retriever_8 = ContextualCompressionRetriever(
    #     base_compressor=compressor_8, base_retriever=pinecone_retriever
    # )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=OPENAI_API_KEY, verbose=True, temperature=0)
    #
    # chain1 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_1,
    #                                               verbose=True)
    # chain2 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_2)
    chain4 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_4)
    # chain8 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_8)


    # for index, question in enumerate(question_list):
    #     print("WAITING")
    #
    #     time.sleep(61)
    #     print(question)
    #     sim_chunks_list = parse_doc_to_list(search_helper.similarity_search(question, k=27))
    #
    #     data[index]["single_context"] = parse_rerank_to_list(1, prompt, sim_chunks_list)
    #     data[index]["two_context"] = parse_rerank_to_list(2, prompt, sim_chunks_list)
    #     data[index]["contexts"] = parse_rerank_to_list(4, prompt, sim_chunks_list)
    #     data[index]["eight_context"] = parse_rerank_to_list(8, prompt, sim_chunks_list)

    result = chain4({"question": prompt, "chat_history": chat_history})

        # result1 = chain1({"question": prompt, "chat_history": chat_history})
        #
        # result2 = chain2({"question": prompt, "chat_history": chat_history})
        #
        # result8 = chain8({"question": prompt, "chat_history": chat_history})

    return result

def get_response_rerank_1024(prompt: str, chat_history: List[Tuple[str, str]]) -> Any:
    pinecone.init(environment=PINECONE_ENVIRONMENT_1024, api_key=PINECONE_1024_INDEX_API_KEY)

    search_helper = Pinecone.from_existing_index(index_name="doc-api-hf-data-index-1024", embedding=embeddings)

    pinecone_retriever = search_helper.as_retriever(search_kwargs={"k": 27})
    # compressor_1 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=1)
    # compressor_2 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=2)
    compressor_4 = CohereRerank(cohere_api_key=COHERE_API_KEY, top_n=4)
    # compressor_8 = CohereRerank(cohere_api_key="BQXYQsXawo6tfRYBxbiegFVvmVlQZTuoM2aP4S6Z", top_n=8)
    # openai_compression_retriever_1 = ContextualCompressionRetriever(
    #     base_compressor=compressor_1, base_retriever=pinecone_retriever
    # )
    # openai_compression_retriever_2 = ContextualCompressionRetriever(
    #     base_compressor=compressor_2, base_retriever=pinecone_retriever
    # )
    openai_compression_retriever_4 = ContextualCompressionRetriever(
        base_compressor=compressor_4, base_retriever=pinecone_retriever
    )
    # openai_compression_retriever_8 = ContextualCompressionRetriever(
    #     base_compressor=compressor_8, base_retriever=pinecone_retriever
    # )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo-16k', openai_api_key=OPENAI_API_KEY, verbose=True, temperature=0)

    # chain1 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_1,
    #                                                verbose=True)
    # chain2 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_2)
    chain4 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_4)
    # chain8 = ConversationalRetrievalChain.from_llm(llm=chat, retriever=openai_compression_retriever_8)

    result = chain4({"question": prompt, "chat_history": chat_history})

    # result1 = chain1({"question": prompt, "chat_history": chat_history})
    #
    # result2 = chain2({"question": prompt, "chat_history": chat_history})
    #
    # result8 = chain8({"question": prompt, "chat_history": chat_history})
    return result
