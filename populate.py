import chromadb
from chromadb.utils import embedding_functions
import os
import openai
import json

with open('openaiapikey.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ["OPENAI_API_KEY"] = api_key

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


openai.api_key = open_file('openaiapikey.txt')



chroma_client = chromadb.Client()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai.api_key,
                model_name="text-embedding-ada-002"
            )

collection = chroma_client.create_collection(name="my_collection", embedding_function=openai_ef)


collection.add(
    documents=["Arm Yourself With Specific Knowledge\n\nSpecific knowledge can be found by pursuing your genuine curiosity\n\n6:19\n\nGet podcast\n\nNivi: Do you want to talk a little bit about the skills that you need, in particular specific knowledge, account\n\nability, leverage and judgment. So, the first tweet in this area is \u201cArm yourself with specific knowledge acc\n\nountability and leverage.\u201d And I\u2019ll throw in judgment as well. I don\u2019t think you covered that in that particular\n\ntweet.\n\nNaval: If you want to make money you have to get paid at scale. And why you, that\u2019s accountability, at sc\n\nale, that\u2019s leverage, and just you getting paid as opposed to somebody else getting paid , that\u2019s specific k\n\nnowledge.\n\nSo, specific knowledge is probably the hardest thing to get across in this whole tweetstorm, and it\u2019s proba\n\nbly the thing that people get the most confused about.\n\nThe thing is that we have this idea that everything can be taught, everything can be taught in school. And", "doc lorem ipsum 2", "it\u2019s not true that everything can be taught. In fact, the most interesting things cannot be taught.\n\nBut everything can be learned. And very often that learning either comes from some innate characteristics\n\nin your DNA, or it could be through your childhood where you learn soft skills which are very, very hard to\n\nteach later on in life, or it\u2019s something that is brand new so nobody else knows how to do it either, or it\u2019s tr\n\nue on the job training because you\u2019re pattern matching into highly complex environments, basically buildin\n\ng judgment in a specific domain.\n\nClassic example is investing, but it could be in anything. It could be in judgment in running a fleet of trucks\n\n, it could be judgment in weather forecasting.\n\nSo, specific knowledge is the knowledge that you care about. Especially if you\u2019re later in life, let\u2019s say your\n\npost 20, 21, 22, you almost don\u2019t get to choose which specific knowledge you have. Rather, you get to lo",],
    ids=["id1", "id2", "id3",]
)



