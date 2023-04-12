from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
import os
import nltk
import magic

loader = UnstructuredPDFLoader("sample.pdf")

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data(0).page_content)} characters in your document')

# os.environ["OPENAI_API_KEY"] 
