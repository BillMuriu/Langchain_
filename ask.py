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


# def view_first_three_chunks(texts):
#     return texts[:3]

# first_three_chunks = view_first_three_chunks(texts)
# print(first_three_chunks)

# Assign a unique ID to each chunk
chunks = []
for i, text in enumerate(texts):
    chunk_id = str(uuid.uuid4())
    chunk = {
        "id": chunk_id,
        "text": text
    }
    chunks.append(chunk)

# Store the chunks in a list
print(f'Now you have {len(chunks)} chunks')
for chunk in chunks:
    print(chunk)

