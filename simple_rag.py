import json
import boto3
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AWS_REGION = "ap-south-1"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# Initialize Bedrock Runtime client
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def get_embedding(text: str) -> list[float]:
    """Generate vector embedding for a given string using Amazon Titan Embeddings v2."""
    payload = {
        "inputText": text,
        "dimensions": 512,  # Titan v2 supports 256, 512, or 1024
        "normalize": True,
    }
    response = bedrock_runtime.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two normalized vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SimpleVectorStore:
    """Lightweight in-memory vector store without needing external databases."""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, docs: list[str]):
        print(f"Embedding {len(docs)} document chunks...")
        for doc in docs:
            emb = get_embedding(doc)
            self.documents.append(doc)
            self.embeddings.append(emb)
        print("Documents successfully indexed.\n")

    def search(self, query: str, top_k: int = 2) -> list[tuple[str, float]]:
        query_emb = get_embedding(query)
        scored_docs = [
            (doc, cosine_similarity(query_emb, emb))
            for doc, emb in zip(self.documents, self.embeddings)
        ]
        # Sort descending by similarity score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]


def generate_answer(query: str, retrieved_context: list[str]) -> str:
    """Send retrieved context + user query to Bedrock LLM using the modern Converse API."""
    context_str = "\n\n---\n\n".join(retrieved_context)
    prompt = f"""Use the following context to answer the user's question accurately.
If the answer cannot be found in the context, reply "I cannot find the answer in the provided documents."

Context:
{context_str}

Question:
{query}
"""

    messages = [
        {
            "role": "user",
            "content": [{"text": prompt}],
        }
    ]

    candidate_models = [
        GENERATION_MODEL_ID,
        "amazon.titan-text-express-v1",
        "amazon.nova-micro-v1:0",
    ]

    for model_id in candidate_models:
        try:
            response = bedrock_runtime.converse(
                modelId=model_id,
                messages=messages,
                inferenceConfig={
                    "maxTokens": 512,
                    "temperature": 0.2,
                },
            )
            return response["output"]["message"]["content"][0]["text"]
        except Exception as e:
            if model_id != candidate_models[-1]:
                print(f"[Notice] Model {model_id} error ({e}). Trying fallback...")
            else:
                raise e


def main():
    sample_documents = [
        "Project Nova is an autonomous deep-space probe developed by Nova Spaceworks.",
        "The primary mission objective of Project Nova is exploring ice fissures on Jupiter's moon Europa.",
        "Project Nova launch date is scheduled for October 14, 2027 from Launch Complex 39A.",
        "The probe utilizes a dual NextGen RTG-X2 power plant rated for 18 years at 450 Watts.",
        "Primary propulsion uses the Xenon-Hall Ion Drive Aeon-4 with 4,200 seconds of specific impulse.",
    ]

    vector_store = SimpleVectorStore()
    vector_store.add_documents(sample_documents)

    query = "What is the mission objective of Project Nova and what power system does it use?"
    print(f"User Query: {query}\n")

    # 1. Retrieve
    top_matches = vector_store.search(query, top_k=2)
    print("Retrieved Context:")
    for doc, score in top_matches:
        print(f"- [Score: {score:.4f}] {doc}")
    print()

    # 2. Generate
    context_chunks = [doc for doc, _ in top_matches]
    answer = generate_answer(query, context_chunks)

    print("Bedrock Answer:")
    print(answer)


if __name__ == "__main__":
    main()
