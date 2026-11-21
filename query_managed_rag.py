import json
import sys
from pathlib import Path
import boto3

MANIFEST_FILE = Path(__file__).parent / "managed_rag_manifest.json"


def ask_bedrock(query: str, kb_id: str, region: str):
    agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region)

    model_arns = [
        f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
        f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-text-express-v1",
    ]

    for model_arn in model_arns:
        try:
            response = agent_runtime.retrieve_and_generate(
                input={"text": query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": kb_id,
                        "modelArn": model_arn,
                    },
                },
            )
            return response
        except Exception as e:
            # If Claude 3 is not accessible, fall back to Amazon Titan
            model_short = model_arn.split("/")[-1]
            if model_arn != model_arns[-1]:
                print(f"[Note] Model {model_short} error ({e}). Trying fallback...")
            else:
                raise e


def main():
    if not MANIFEST_FILE.exists():
        print(f"Error: Manifest file '{MANIFEST_FILE.name}' not found.")
        print("Please run 'py -3.12 deploy_managed_rag.py' first.")
        sys.exit(1)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    kb_id = manifest.get("knowledge_base_id")
    region = manifest.get("region", "ap-south-1")

    if not kb_id:
        print("Error: Knowledge Base ID missing from manifest.")
        sys.exit(1)

    print("=" * 70)
    print(f"  Bedrock Knowledge Base Query CLI  (KB ID: {kb_id})")
    print("  Type your question below, or 'exit' / 'quit' to leave.")
    print("=" * 70)

    # Allow query via command line arg or interactive loop
    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        queries = None

    while True:
        if queries:
            query = queries.pop(0)
        else:
            try:
                query = input("\nEnter Question: ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        print(f"\nSearching Knowledge Base & Generating Answer...")
        try:
            resp = ask_bedrock(query, kb_id, region)
            answer = resp["output"]["text"]
            citations = resp.get("citations", [])

            print("\n" + "-" * 70)
            print("BEDROCK ANSWER:")
            print("-" * 70)
            print(answer)

            if citations:
                print("\nCITATIONS & RETRIEVED SOURCES:")
                for cit in citations:
                    for ref in cit.get("retrievedReferences", []):
                        s3_uri = ref.get("location", {}).get("s3Location", {}).get("uri", "S3")
                        snippet = ref.get("content", {}).get("text", "").replace("\n", " ")
                        print(f"  • Source: {s3_uri}")
                        print(f"    Text: {snippet[:200]}...")
            print("-" * 70)
        except Exception as e:
            print(f"Error during retrieve-and-generate: {e}")

        if queries is not None and len(queries) == 0:
            break


if __name__ == "__main__":
    main()
