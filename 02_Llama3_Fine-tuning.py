# Databricks notebook source
# MAGIC %md
# MAGIC ## 環境：Runtime 15.4 LTS ML - Single Node

# COMMAND ----------

!pip install -U mlflow
!pip install -U transformers
!pip install -U accelerate
!pip install -U tf-keras
!pip install -U langchain
!pip install -U langchain_community
!pip install databricks_genai

dbutils.library.restartPython()

# COMMAND ----------

from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
import pandas as pd
import mlflow
from databricks.model_training import foundation_model as fm
from mlflow.tracking import MlflowClient

# COMMAND ----------

catalog = "YOUR-CATALOG-NAME"
schema = "YOUR-SCHEMA-NAME"
model_schema = "YOUR-MODEL-SCHEMA-NAME"
volume = "YOUR-VOLUME-NAME"

# COMMAND ----------

mlflow.set_registry_uri('databricks-uc')

# COMMAND ----------

# DBTITLE 1,スキーマとボリューム作成
# スキーマがなければ作成
# dataset
create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}"
spark.sql(create_schema_query)

# model
create_schema_query = f"CREATE SCHEMA IF NOT EXISTS {catalog}.{model_schema}"
spark.sql(create_schema_query)

# ボリュームがなければ作成
create_volume_query = f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}"
spark.sql(create_volume_query)

# COMMAND ----------

import pandas as pd 

# JSONLファイルを読んで、Delta Tableとして保存するコード
bronze_table_path = f"{catalog}.{schema}.bronze_train_data"

# Drop if table existing
sql(f"drop table if exists {bronze_table_path}")

unity_catalog_volume_path = './sample_data/raft_training_dataset_ja.jsonl'

pandasDF = pd.read_json(path_or_buf=unity_catalog_volume_path, lines=True)
sparkDF=spark.createDataFrame(pandasDF)

# Delta Tableとして保存する
sparkDF.write.mode("overwrite").saveAsTable(bronze_table_path)
display(sparkDF)

# COMMAND ----------

training_dataset = (
    spark.table(bronze_table_path)
    .select("instruction","cot_answer")
    .withColumnRenamed("cot_answer", "response")
    .withColumnRenamed("instruction", "prompt")
)

silver_table_path = f"{catalog}.{schema}.silver_train_data"
training_dataset.write.mode("overwrite").saveAsTable(silver_table_path)
training_dataset.display()

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd

base_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

system_prompt = """* You are an excellent AI assistant.
* Please respond to user questions based on the reference information enclosed within  <DOCUMENT> and </DOCUMENT>. However, some of the reference information may be irrelevant, so please ignore any unrelated parts.
* Please provide your response in Japanese."""

@pandas_udf("array<struct<role:string, content:string>>")
def create_conversation(question: pd.Series, answer: pd.Series) -> pd.Series:
    def build_message(q,a):
        user_input = f"please write an SQL query to answer the following question: {q}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": a}]
    return pd.Series([build_message(q, a) for q, a in zip(question, answer)])

gold_table_path = f"{catalog}.{schema}.chat_completion_training_dataset"
training_dataset.select(create_conversation("prompt", "response").alias('messages')).write.mode('overwrite').saveAsTable(gold_table_path)

display(spark.table(gold_table_path))

# COMMAND ----------

import re
from databricks.model_training import foundation_model as fm

# データセットを読み取り、ファインチューニングクラスターに送信するために使う現行のクラスターのIDを返却。https://docs.databricks.com/en/large-language-models/foundation-model-training/create-fine-tune-run.html#cluster-id をご覧ください
def get_current_cluster_id():
  import json
  return json.loads(dbutils.notebook.entry_point.getDbutils().notebook().getContext().safeToJson())['attributes']['clusterId']

registered_model_name = f"{catalog}.{model_schema}.llama31_8b_ift"

run = fm.create(
    data_prep_cluster_id=get_current_cluster_id(),  # トレーニングデータソースとしてDeltaテーブルを使っている際には必須。データ準備で使用するクラスターID。
    model=base_model_name,
    train_data_path=gold_table_path,
    task_type="CHAT_COMPLETION", 
    register_to=registered_model_name,
    training_duration="2ep",
    # learning_rate="5e-7",
)

print(f"Fine-tuning run details: {run}")
