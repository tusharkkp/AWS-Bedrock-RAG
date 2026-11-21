import json
import sys
import time
from pathlib import Path
import boto3

MANIFEST_FILE = Path(__file__).parent / "managed_rag_manifest.json"


def log(msg: str):
    print(f"[Ingestion] {msg}")


def main():
    if not MANIFEST_FILE.exists():
        log(f"Error: Manifest '{MANIFEST_FILE.name}' not found.")
        sys.exit(1)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    region = manifest.get("region", "ap-south-1")
    kb_id = manifest.get("knowledge_base_id")
    ds_id = manifest.get("data_source_id")

    if not kb_id or not ds_id:
        log("Error: Knowledge Base ID or Data Source ID is missing from manifest.")
        sys.exit(1)

    bedrock_agent = boto3.client("bedrock-agent", region_name=region)
    agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region)

    log(f"Starting Ingestion Job for KB '{kb_id}' and Data Source '{ds_id}' in '{region}'...")
    try:
        job_resp = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            description="Document ingestion",
        )
        job_id = job_resp["ingestionJob"]["ingestionJobId"]
        log(f"Job started! Ingestion Job ID: {job_id}")

        while True:
            job_status = bedrock_agent.get_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=ds_id,
                ingestionJobId=job_id,
            )["ingestionJob"]["status"]
            log(f"Ingestion job status: {job_status}")
            if job_status == "COMPLETE":
                log("Ingestion completed successfully!")
                break
            elif job_status in ("FAILED", "STOPPED"):
                raise RuntimeError(f"Ingestion job failed with status: {job_status}")
            time.sleep(5)

        print("\n" + "=" * 70)
        print("  INGESTION COMPLETE! Testing Knowledge Base query...")
        print("=" * 70)
        test_query = "What is the mission objective of Project Nova and when is the launch date?"
        print(f"Query: {test_query}\n")

        # Try Nova or Claude
        for model_arn in [
            f"arn:aws:bedrock:{region}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
            f"arn:aws:bedrock:{region}::foundation-model/amazon.titan-text-express-v1",
        ]:
            try:
                rag_resp = agent_runtime.retrieve_and_generate(
                    input={"text": test_query},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": kb_id,
                            "modelArn": model_arn,
                        },
                    },
                )
                print("Answer from Bedrock:")
                print(rag_resp["output"]["text"])
                break
            except Exception as e:
                pass

    except Exception as e:
        log(f"Ingestion failed: {e}")


if __name__ == "__main__":
    main()
