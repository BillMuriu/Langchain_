import pinecone
import json

# with open("chunks.json", "r") as f:
#     chunks = [json.loads(line) for line in f]

# for i in range(3):
#     chunk = chunks[i]
#     print(f"Chunk {i + 1}")
#     print(f"ID: {chunk['id']}")
#     print(f"Text: {chunk['text']}")
#     print(f"Embedding: {chunk['embedding']}")
#     print()

pinecone.init(
    api_key = "0c54847e-ce3a-4eef-9d26-e19e6f93a35c",
    environment = "asia-southeast1-gcp"
)

index_name = "first-index"


index = pinecone.Index("first-index")
