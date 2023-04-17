import chromadb
from chromadb.utils import embedding_functions
import os
import openai

with open('openaiapikey.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')



chroma_client = chromadb.Client()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai.api_key,
                model_name="text-embedding-ada-002"
            )

collection = chroma_client.create_collection(name="my_collection", embedding_function=openai_ef)