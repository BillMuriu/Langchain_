import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import os
import openai
import json


client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="/Users/USER/Documents/gpt3-tut/Gpt3_index_/.chroma" # Optional, defaults to .chromadb/ in the current directory
))

with open('openaiapikey.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai.api_key,
                model_name="text-embedding-ada-002"
            )

# collection = client.create_collection(name="my_collection", embedding_function=openai_ef)
collection = client.get_collection(name="my_collection", embedding_function=openai_ef)
# print(collection)


# with open("chunks.json", "r") as f:
#     chunks = [json.loads(line) for line in f]

# documents = []
# embeddings = []
# ids = []

# for chunk in chunks:
#     documents.append(chunk["text"].strip())
#     embeddings.append(chunk["embedding"])
#     ids.append(chunk["id"])

# collection.add(documents=documents, embeddings=embeddings, ids=ids)

results = collection.query(
    query_texts=["What types of leverage should I focus on acquiring?"],
    n_results=3
)


document = results['documents'][0][0]

prompt = f"summarize the following text in one paragraph(200 word paragraph). Ignore the words and parts that seem incomplete. {document}"


response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompt,
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response.choices[0].text)