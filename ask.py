from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os
import nltk
import magic

os.environ["OPENAI_API_KEY"] = 

loader = DirectoryLoader('chunk_txt/', glob='**/*.txt')
docs = loader.load()

char_text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap = 0)

doc_chunks = char_text_splitter.split_documents(docs)

print(doc_chunks)

openAI_embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

embeddings = openAI_embeddings.embed_documents(texts=doc_chunks)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

# Extract only the embedded vectors from each document
vector_list = []
for doc_vectors in embeddings:
    for vector in doc_vectors:
        vector_list.append(vector)

vstore = Chroma.from_vectors(vector_list)
