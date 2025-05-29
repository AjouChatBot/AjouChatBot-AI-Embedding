import pandas as pd

from datetime import datetime
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import OpenAI

from Milvus_embedding import process_json_item
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dotenv import load_dotenv

load_dotenv()

# Milvus 연결
connections.connect(
    host=os.getenv("MILVUS_HOST"),
    port=os.getenv("MILVUS_PORT")
)
# Milvus 컬렉션 스키마 정의
collection_name = "a_mate"
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    FieldSchema(name="createAt", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="scrapUrl", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="urlTitle", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="dataType", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="fileName", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="fileIndex", dtype=DataType.VARCHAR, max_length=50),
    FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=1000)
]

schema = CollectionSchema(fields, description="Embeddings with metadata")
main_collection = Collection(name=collection_name, schema=schema)

keyword_collection_name = "a_mate_keywords"
keyword_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="keyword", dtype=DataType.VARCHAR, max_length=200)
]
keyword_schema = CollectionSchema(keyword_fields, description="Keyword-only embeddings")

if not utility.has_collection(keyword_collection_name):
    keyword_collection = Collection(name=keyword_collection_name, schema=keyword_schema)
else:
    keyword_collection = Collection(name=keyword_collection_name)

# OpenAI API 설정
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def text_embedding_from_excel(file_path: str):
    df = pd.read_excel(file_path)

    # 첫 번째 열 = 텍스트, 두 번째 열 = 카테고리
    text_col = df.columns[0]
    category_col = df.columns[1]

    total_chunks = 0

    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        category = str(row[category_col]).strip() or "default"

        if not text:
            continue

        partition_name = category.replace("-", "_")

        if not main_collection.has_partition(partition_name):
            main_collection.create_partition(partition_name)

        item = {
            "text": text,
            "createAt": datetime.now().isoformat(),
            "scrapUrl": "",
            "urlTitle": "",
            "dataType": "excel",
            "fileName": os.path.basename(file_path),
            "fileIndex": str(idx)
        }

        records = process_json_item(item)
        total_chunks += len(records[0])
        main_collection.insert(records, partition_name=partition_name)

        logging.info(f"[Row {idx}] Inserted {len(records[0])} chunks into partition '{partition_name}'")

    main_collection.flush()
    logging.info(f"✅ Embedding complete. Total {total_chunks} chunks inserted from Excel.")

if __name__ == "__main__":
    # 예: 엑셀 파일 경로 전달
    text_embedding_from_excel("data/data.xlsx")