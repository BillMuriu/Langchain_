from langchain.embeddings.openai import OpenAIEmbedding
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA


openai = OpenAIEmbeddings(openai_api_key="my-api-key")