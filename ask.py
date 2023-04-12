from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.vectorstores import Pinecone
import pinecone
import os
import nltk
import magic

with open('openaiapikey.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key


loader = UnstructuredPDFLoader("sample.pdf")

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print(f'There are {len(data[0].page_content)} characters in your document')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])


# initialize pinecone

pinecone.init(
    
)

index_name = "first-index"

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

query = "What is Specific Knowledge?"
docs = docsearch.similarity_search(query, include_metadata=True)

for doc in docs:
    print(doc.page_content.encode('utf-8'))