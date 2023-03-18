from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.embeddings.openai import OpenAIEmbeddings
import magic
import os
import nltk

os.environ['OPENAI_API_KEY'] =''

loader = DirectoryLoader('/chunk_txt', glob='**/*.txt')