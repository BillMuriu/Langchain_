from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import nltk
import magic
import json
import uuid

with open('openaiapikey.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key


loader = UnstructuredPDFLoader("sample.pdf")

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')


chunks = []
for i, text in enumerate(texts):
    chunk_id = str(uuid.uuid4())
    chunk = {
        "id": chunk_id,
        "text": text
    }
    chunks.append(chunk)


for i, chunk in enumerate(chunks):
    if i < 3:
        print(f"Chunk ID: {i+1}")
        print(chunk)


# create a list to hold the chunk dictionaries
chunk_dicts = []
for i, text in enumerate(texts):
    chunk_id = str(uuid.uuid4())
    chunk = {
        "id": chunk_id,
        "text": text
    }
    chunk_dicts.append(chunk)

with open("chunks.json", "w") as f:
    for chunk_dict in chunk_dicts:
        chunk_dict["text"] = chunk_dict["text"].__dict__  # Convert Document object to dictionary
        json.dump(chunk_dict, f)
        f.write("\n")
