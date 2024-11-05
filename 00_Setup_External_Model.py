# Databricks notebook source
# MAGIC %md
# MAGIC ## 環境：サーバーレス・ノートブック

# COMMAND ----------

# MAGIC %pip install mlflow

# COMMAND ----------

import mlflow.deployments
client = mlflow.deployments.get_deploy_client("databricks")

# COMMAND ----------

AZURE_OPENAI_BASE = "YOUR_AZURE_OPENAI_BASE"
LLM_ENDPOINT_NAME = "YOUR-LLM-ENDPOINT-NAME"
EMBEDDING_ENDPOINT_NAME = "YOUR-EMBEDDING-ENDPOINT-NAME"

# COMMAND ----------

client.create_endpoint(
    name=LLM_ENDPOINT_NAME,
    config={
        "served_entities": [{
            "external_model": {
                "name": "gpt-4o",
                "provider": "openai",
                "task": "llm/v1/chat",
                "openai_config": {
                    "openai_api_type": "azure",
                    "openai_api_key": "{{secrets/hiouchiy/azure_openai_token}}",
                    "openai_api_base": AZURE_OPENAI_BASE,
                    "openai_deployment_name": "gpt-4o",
                    "openai_api_version": "2024-08-01-preview"
                }
            }
        }]
    }
)

# COMMAND ----------

client.create_endpoint(
    name=EMBEDDING_ENDPOINT_NAME,
    config={
        "served_entities": [{
            "external_model": {
                "name": "text-embedding-ada-002",
                "provider": "openai",
                "task": "llm/v1/embeddings",
                "openai_config": {
                    "openai_api_type": "azure",
                    "openai_api_key": "{{secrets/hiouchiy/azure_openai_token}}",
                    "openai_api_base": AZURE_OPENAI_BASE,
                    "openai_deployment_name": "text-embedding-ada-002",
                    "openai_api_version": "2023-05-15"
                }
            }
        }]
    }
)

# COMMAND ----------

client.delete_endpoint(LLM_ENDPOINT_NAME)
client.delete_endpoint(EMBEDDING_ENDPOINT_NAME)
