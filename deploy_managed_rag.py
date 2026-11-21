import json
import time
import uuid
import sys
from pathlib import Path
import boto3

REGION = "ap-south-1"
EMBEDDING_MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/amazon.titan-embed-text-v2:0"
# Generation model for Bedrock Knowledge Base retrieve-and-generate
GENERATION_MODEL_ARN = f"arn:aws:bedrock:{REGION}::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"

MANIFEST_FILE = Path(__file__).parent / "managed_rag_manifest.json"
SAMPLE_DOC = Path(__file__).parent / "sample_docs" / "nova_spaceworks_handbook.txt"


def log(msg: str):
    print(f"[Managed-RAG] {msg}")


def main():
    print("=" * 70)
    print("  AWS Bedrock Managed Knowledge Base (Approach 2) - Automated Setup")
    print("=" * 70)

    # 1. Verify AWS Authentication
    sts = boto3.client("sts", region_name=REGION)
    try:
        caller = sts.get_caller_identity()
        account_id = caller["Account"]
        caller_arn = caller["Arn"]
        log(f"Authenticated as: {caller_arn} (Account: {account_id})")
    except Exception as e:
        log(f"ERROR: AWS credentials are not configured or invalid.")
        log(f"Details: {e}")
        log("Please run 'aws configure' in your terminal and try again.")
        sys.exit(1)

    # Resolve IAM principal for caller (convert assumed-role to role ARN if needed)
    if ":assumed-role/" in caller_arn:
        role_name = caller_arn.split("/")[1]
        principal_caller_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
    else:
        principal_caller_arn = caller_arn

    # Generate unique identifiers
    suffix = uuid.uuid4().hex[:6]
    bucket_name = f"bedrock-rag-demo-{account_id}-{suffix}"
    role_name = f"BedrockKBRole-{suffix}"
    policy_name = f"BedrockKBPolicy-{suffix}"
    aoss_col_name = f"rag-col-{suffix}"
    aoss_enc_policy = f"rag-enc-{suffix}"
    aoss_net_policy = f"rag-net-{suffix}"
    aoss_acc_policy = f"rag-acc-{suffix}"
    aoss_index_name = "bedrock-knowledge-base-index"
    kb_name = f"managed-rag-kb-{suffix}"
    ds_name = f"s3-docs-{suffix}"

    s3 = boto3.client("s3", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)
    aoss = boto3.client("opensearchserverless", region_name=REGION)
    bedrock_agent = boto3.client("bedrock-agent", region_name=REGION)

    manifest = {
        "region": REGION,
        "account_id": account_id,
        "suffix": suffix,
        "bucket_name": bucket_name,
        "role_name": role_name,
        "policy_name": policy_name,
        "aoss_collection_name": aoss_col_name,
        "aoss_collection_id": None,
        "aoss_collection_arn": None,
        "aoss_index_name": aoss_index_name,
        "aoss_enc_policy": aoss_enc_policy,
        "aoss_net_policy": aoss_net_policy,
        "aoss_acc_policy": aoss_acc_policy,
        "knowledge_base_id": None,
        "knowledge_base_name": kb_name,
        "data_source_id": None,
        "data_source_name": ds_name,
    }

    try:
        # -------------------------------------------------------------
        # STEP 1: Create S3 Bucket and Upload Document
        # -------------------------------------------------------------
        log(f"1/7 Creating S3 bucket '{bucket_name}'...")
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )

        if not SAMPLE_DOC.exists():
            raise FileNotFoundError(f"Sample doc not found at {SAMPLE_DOC}")

        log(f"Uploading '{SAMPLE_DOC.name}' to S3...")
        s3.upload_file(str(SAMPLE_DOC), bucket_name, f"docs/{SAMPLE_DOC.name}")
        log("S3 bucket ready and sample document uploaded.")

        # -------------------------------------------------------------
        # STEP 2: Create IAM Role for Bedrock Knowledge Base
        # -------------------------------------------------------------
        log(f"2/7 Creating IAM role '{role_name}' for Bedrock...")
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role_resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="Execution role for Amazon Bedrock Knowledge Base",
        )
        kb_role_arn = role_resp["Role"]["Arn"]
        manifest["kb_role_arn"] = kb_role_arn
        log(f"Created IAM role ARN: {kb_role_arn}")

        # -------------------------------------------------------------
        # STEP 3: Setup OpenSearch Serverless Security Policies
        # -------------------------------------------------------------
        log(f"3/7 Setting up OpenSearch Serverless (AOSS) security policies...")

        # Encryption Policy
        enc_policy_doc = {
            "Rules": [{"ResourceType": "collection", "Resource": [f"collection/{aoss_col_name}"]}] ,
            "AWSOwnedKey": True,
        }
        aoss.create_security_policy(
            name=aoss_enc_policy,
            type="encryption",
            policy=json.dumps(enc_policy_doc),
        )

        # Network Policy (Public access so your client & Bedrock can reach it)
        net_policy_doc = [
            {
                "Rules": [
                    {"ResourceType": "collection", "Resource": [f"collection/{aoss_col_name}"]},
                    {"ResourceType": "dashboard", "Resource": [f"collection/{aoss_col_name}"]},
                ],
                "AllowFromPublic": True,
            }
        ]
        aoss.create_security_policy(
            name=aoss_net_policy,
            type="network",
            policy=json.dumps(net_policy_doc),
        )

        # Data Access Policy (Grants rights to caller & Bedrock KB role)
        acc_policy_doc = [
            {
                "Rules": [
                    {
                        "ResourceType": "collection",
                        "Resource": [f"collection/{aoss_col_name}"],
                        "Permission": [
                            "aoss:CreateCollectionItems",
                            "aoss:DeleteCollectionItems",
                            "aoss:UpdateCollectionItems",
                            "aoss:DescribeCollectionItems",
                        ],
                    },
                    {
                        "ResourceType": "index",
                        "Resource": [f"index/{aoss_col_name}/*"],
                        "Permission": [
                            "aoss:CreateIndex",
                            "aoss:DeleteIndex",
                            "aoss:UpdateIndex",
                            "aoss:DescribeIndex",
                            "aoss:ReadDocument",
                            "aoss:WriteDocument",
                        ],
                    },
                ],
                "Principal": [principal_caller_arn, kb_role_arn],
                "Description": "Data access policy for Bedrock Knowledge Base",
            }
        ]
        aoss.create_access_policy(
            name=aoss_acc_policy,
            type="data",
            policy=json.dumps(acc_policy_doc),
        )

        # -------------------------------------------------------------
        # STEP 4: Create OpenSearch Serverless Collection & Vector Index
        # -------------------------------------------------------------
        log(f"4/7 Creating OpenSearch Serverless vector collection '{aoss_col_name}'...")
        col_resp = aoss.create_collection(name=aoss_col_name, type="VECTORSEARCH")
        col_id = col_resp["createCollectionDetail"]["id"]
        manifest["aoss_collection_id"] = col_id

        log("Waiting for OpenSearch Serverless collection to become ACTIVE (~60-90s)...")
        while True:
            desc = aoss.batch_get_collection(ids=[col_id])["collectionDetails"][0]
            status = desc["status"]
            if status == "ACTIVE":
                col_arn = desc["arn"]
                manifest["aoss_collection_arn"] = col_arn
                log(f"Collection is ACTIVE! ARN: {col_arn}")
                break
            elif status == "FAILED":
                raise RuntimeError(f"OpenSearch Serverless collection failed to create.")
            time.sleep(10)

        # Wait an extra 10 seconds for AOSS security policies to propagate
        time.sleep(10)

        log(f"Creating vector index '{aoss_index_name}' in collection...")
        index_schema = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "bedrock-knowledge-base-default-vector": {
                        "type": "knn_vector",
                        "dimension": 1024,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "l2",
                        },
                    },
                    "AMAZON_BEDROCK_TEXT_CHUNK": {"type": "text", "index": True},
                    "AMAZON_BEDROCK_METADATA": {"type": "text", "index": False},
                }
            },
        }

        aoss.create_index(
            id=col_id,
            indexName=aoss_index_name,
            indexSchema=index_schema,
        )
        log("Vector index created successfully.")

        # Attach permissions policy to IAM Role
        log("Attaching permissions policy to Bedrock execution role...")
        kb_policy_doc = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["bedrock:InvokeModel"],
                    "Resource": EMBEDDING_MODEL_ARN,
                },
                {
                    "Effect": "Allow",
                    "Action": ["aoss:APIAccessAll"],
                    "Resource": col_arn,
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket_name}",
                        f"arn:aws:s3:::{bucket_name}/*",
                    ],
                },
            ],
        }
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName=policy_name,
            PolicyDocument=json.dumps(kb_policy_doc),
        )
        log("Sleeping 15 seconds for IAM and OpenSearch replication...")
        time.sleep(15)

        # -------------------------------------------------------------
        # STEP 5: Create Bedrock Knowledge Base
        # -------------------------------------------------------------
        log(f"5/7 Creating Bedrock Knowledge Base '{kb_name}'...")
        kb_config = {
            "type": "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": EMBEDDING_MODEL_ARN
            },
        }
        storage_config = {
            "type": "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": {
                "collectionArn": col_arn,
                "vectorIndexName": aoss_index_name,
                "fieldMapping": {
                    "vectorField": "bedrock-knowledge-base-default-vector",
                    "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
                    "metadataField": "AMAZON_BEDROCK_METADATA",
                },
            },
        }

        # Retry loop for eventual consistency of IAM role trust policy
        for attempt in range(5):
            try:
                kb_resp = bedrock_agent.create_knowledge_base(
                    name=kb_name,
                    description="Bedrock RAG Managed Knowledge Base Demo",
                    roleArn=kb_role_arn,
                    knowledgeBaseConfiguration=kb_config,
                    storageConfiguration=storage_config,
                )
                kb_id = kb_resp["knowledgeBase"]["knowledgeBaseId"]
                manifest["knowledge_base_id"] = kb_id
                log(f"Knowledge Base created! ID: {kb_id}")
                break
            except bedrock_agent.exceptions.AccessDeniedException as e:
                log(f"IAM role still propagating (attempt {attempt+1}/5). Waiting 10s...")
                time.sleep(10)

        # Wait until KB is ACTIVE
        while True:
            kb_status = bedrock_agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"]
            if kb_status == "ACTIVE":
                log("Knowledge Base status is ACTIVE.")
                break
            time.sleep(5)

        # -------------------------------------------------------------
        # STEP 6: Create Data Source & Start Ingestion
        # -------------------------------------------------------------
        log(f"6/7 Creating S3 Data Source '{ds_name}'...")
        ds_resp = bedrock_agent.create_data_source(
            knowledgeBaseId=kb_id,
            name=ds_name,
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{bucket_name}",
                    "inclusionPrefixes": ["docs/"],
                },
            },
        )
        ds_id = ds_resp["dataSource"]["dataSourceId"]
        manifest["data_source_id"] = ds_id
        log(f"Data Source created! ID: {ds_id}")

        log("Starting Ingestion Job (chunking, embedding, indexing)...")
        job_resp = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            description="Initial document ingestion",
        )
        job_id = job_resp["ingestionJob"]["ingestionJobId"]

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
                raise RuntimeError(f"Ingestion job ended with status: {job_status}")
            time.sleep(5)

        # Save manifest
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log(f"Saved deployment manifest to '{MANIFEST_FILE.name}'.")

        # -------------------------------------------------------------
        # STEP 7: Run Initial Test Query
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("  Testing Bedrock RetrieveAndGenerate API")
        print("=" * 70)
        test_query = "What is the mission objective of Project Nova and when is the launch date?"
        print(f"Query: {test_query}\n")

        agent_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)
        try:
            rag_response = agent_runtime.retrieve_and_generate(
                input={"text": test_query},
                retrieveAndGenerateConfiguration={
                    "type": "KNOWLEDGE_BASE",
                    "knowledgeBaseConfiguration": {
                        "knowledgeBaseId": kb_id,
                        "modelArn": GENERATION_MODEL_ARN,
                    },
                },
            )
            print("Answer from Bedrock:")
            print(rag_response["output"]["text"])
            print("\nCitations:")
            for cit in rag_response.get("citations", []):
                for ref in cit.get("retrievedReferences", []):
                    s3_uri = ref.get("location", {}).get("s3Location", {}).get("uri", "")
                    content = ref.get("content", {}).get("text", "")
                    print(f"- Source: {s3_uri}")
                    print(f"  Excerpt: {content[:150]}...")
        except Exception as e:
            log(f"Test query note: {e}")
            log("You can run 'py -3.12 query_managed_rag.py' to query once model access is active.")

        print("\n" + "=" * 70)
        print("  DEPLOYMENT COMPLETE!")
        print(f"  Knowledge Base ID: {kb_id}")
        print("  To ask more questions, run:  py -3.12 query_managed_rag.py")
        print("  To DELETE ALL resources, run: py -3.12 teardown_managed_rag.py")
        print("=" * 70)

    except Exception as e:
        log(f"Deployment encountered an error: {e}")
        # Save whatever was created so teardown can clean it up
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        log("Manifest saved. Run 'py -3.12 teardown_managed_rag.py' to clean up partial resources.")
        raise e


if __name__ == "__main__":
    main()
