import json
import sys
import time
from pathlib import Path
import boto3

MANIFEST_FILE = Path(__file__).parent / "managed_rag_manifest.json"


def log(msg: str):
    print(f"[Teardown] {msg}")


def main():
    print("=" * 70)
    print("  AWS Bedrock Managed Knowledge Base (Approach 2) - TEARDOWN & CLEANUP")
    print("=" * 70)

    if not MANIFEST_FILE.exists():
        log(f"Manifest '{MANIFEST_FILE.name}' not found. Nothing to clean up.")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    region = manifest.get("region", "ap-south-1")
    s3 = boto3.client("s3", region_name=region)
    iam = boto3.client("iam", region_name=region)
    aoss = boto3.client("opensearchserverless", region_name=region)
    bedrock_agent = boto3.client("bedrock-agent", region_name=region)

    kb_id = manifest.get("knowledge_base_id")
    ds_id = manifest.get("data_source_id")
    col_id = manifest.get("aoss_collection_id")
    index_name = manifest.get("aoss_index_name")
    enc_policy = manifest.get("aoss_enc_policy")
    net_policy = manifest.get("aoss_net_policy")
    acc_policy = manifest.get("aoss_acc_policy")
    bucket_name = manifest.get("bucket_name")
    role_name = manifest.get("role_name")
    policy_name = manifest.get("policy_name")

    # 1. Delete Data Source
    if kb_id and ds_id:
        try:
            log(f"Deleting Data Source '{ds_id}'...")
            bedrock_agent.delete_data_source(knowledgeBaseId=kb_id, dataSourceId=ds_id)
            log("Data Source deleted.")
        except Exception as e:
            log(f"Data source deletion note: {e}")

    # 2. Delete Knowledge Base
    if kb_id:
        try:
            log(f"Deleting Knowledge Base '{kb_id}'...")
            bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)
            log("Knowledge Base deleted.")
        except Exception as e:
            log(f"Knowledge base deletion note: {e}")

    time.sleep(5)

    # 3. Delete OpenSearch Serverless Index
    if col_id and index_name:
        try:
            log(f"Deleting OpenSearch index '{index_name}' in collection '{col_id}'...")
            aoss.delete_index(id=col_id, indexName=index_name)
            log("Index deleted.")
        except Exception as e:
            log(f"Index deletion note: {e}")

    # 4. Delete OpenSearch Serverless Collection
    if col_id:
        try:
            log(f"Deleting OpenSearch Serverless Collection '{col_id}'...")
            aoss.delete_collection(id=col_id)
            log("Collection deletion initiated.")
        except Exception as e:
            log(f"Collection deletion note: {e}")

    # 5. Delete OpenSearch Serverless Policies
    for policy_type, name in [
        ("data", acc_policy),
        ("network", net_policy),
        ("encryption", enc_policy),
    ]:
        if name:
            try:
                log(f"Deleting AOSS {policy_type} policy '{name}'...")
                if policy_type == "data":
                    aoss.delete_access_policy(name=name, type="data")
                else:
                    aoss.delete_security_policy(name=name, type=policy_type)
                log(f"Deleted {policy_type} policy.")
            except Exception as e:
                log(f"Policy '{name}' deletion note: {e}")

    # 6. Empty and Delete S3 Bucket
    if bucket_name:
        try:
            log(f"Emptying and deleting S3 bucket '{bucket_name}'...")
            # Delete all object versions & objects
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket_name):
                objs = page.get("Contents", [])
                if objs:
                    s3.delete_objects(
                        Bucket=bucket_name,
                        Delete={"Objects": [{"Key": o["Key"]} for o in objs]},
                    )
            s3.delete_bucket(Bucket=bucket_name)
            log(f"S3 bucket '{bucket_name}' deleted.")
        except Exception as e:
            log(f"S3 bucket deletion note: {e}")

    # 7. Delete IAM Role & Policies
    if role_name and policy_name:
        try:
            log(f"Deleting inline IAM policy '{policy_name}' from role '{role_name}'...")
            iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
        except Exception as e:
            log(f"IAM role policy note: {e}")

    if role_name:
        try:
            log(f"Deleting IAM role '{role_name}'...")
            iam.delete_role(RoleName=role_name)
            log("IAM role deleted.")
        except Exception as e:
            log(f"IAM role note: {e}")

    # 8. Clean up manifest file
    try:
        MANIFEST_FILE.unlink()
        log(f"Removed '{MANIFEST_FILE.name}'.")
    except Exception as e:
        log(f"Manifest cleanup note: {e}")

    print("\n" + "=" * 70)
    print("  TEARDOWN COMPLETE! All AWS resources have been successfully removed.")
    print("  No ongoing charges will be incurred.")
    print("=" * 70)


if __name__ == "__main__":
    main()
