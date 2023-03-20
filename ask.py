from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

os.environ["OPENAI_API_KEY"] = ''

def construct_index():
    max_input_size = 4096
    num_outputs = 256
    max_chunk_overlap = 20
    chunk_size_limit = 600


    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit)

    