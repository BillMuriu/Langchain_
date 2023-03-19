from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
import magic
import os
import nltk

os.environ['OPENAI_API_KEY'] ='

loader = DirectoryLoader('/chunk_txt', glob='**/*.txt')


documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)


# texts = text_splitter.split_documents(documents)


# embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])


# docsearch = Chroma.from_documents(texts, embeddings)


# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)


# query = "What is Naval Ravikant's Story?"
# qa.run(query)