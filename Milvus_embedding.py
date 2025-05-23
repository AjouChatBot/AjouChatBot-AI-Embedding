from datetime import datetime
import pymysql
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from KeyWord_Extractor import extract_keywords
import os
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dotenv import load_dotenv

load_dotenv()

# Milvus 연결
connections.connect(host='mate.ajou.app', port='28116')

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

# MySQL에서 데이터 불러오기
def load_data_from_mysql():
    conn = pymysql.connect(
        host="mate.ajou.app",
        port=28115,
        user="amate_admin",
        password="Amate2025*",
        db="amate",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT c.id, content_text AS text, c.created_at AS createAt,
                       scrap_url AS scrapUrl, url_title AS urlTitle, c.data_type AS dataType, 
                       org_file_name AS fileName, file_ref_index AS fileIndex
                FROM contents c
                LEFT JOIN scrap_info si ON c.id = si.content_id
            """
            cursor.execute(sql)
            results = cursor.fetchall()
            data_type_map = {0: "text", 1: "img", 2: "document"}
            for row in results:
                if "createAt" in row:
                    row["createAt"] = row["createAt"].isoformat() if isinstance(row["createAt"], (datetime,)) else row["createAt"]
                if "dataType" in row:
                    row["dataType"] = data_type_map.get(row["dataType"], str(row["dataType"]))
                row["category"] = "Facilities"
            return results
    finally:
        conn.close()

# 텍스트 청킹 및 임베딩
def process_json_item(item: dict):
    text = item.get("text", "")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_text(text)

    metadata_base = {
        "createAt": str(item.get("createAt", "")) if item.get("createAt") is not None else "",
        "scrapUrl": str(item.get("scrapUrl", "")) if item.get("scrapUrl") is not None else "",
        "urlTitle": str(item.get("urlTitle", "")) if item.get("urlTitle") is not None else "",
        "dataType": str(item.get("dataType", "")) if item.get("dataType") is not None else "",
        "fileName": str(item.get("fileName", "")) if item.get("fileName") is not None else "",
        "fileIndex": str(item.get("fileIndex", "")) if item.get("fileIndex") is not None else ""
    }

    insert_data = []
    for chunk in chunks:
        embedding = openai.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding
        keywords = extract_keywords(chunk)
        ## 추출된 키워드들을 키워드 데이터베이스에 저장
        for keyword in keywords:
            keyword_embedding = openai.embeddings.create(input=keyword, model="text-embedding-ada-002").data[
                0].embedding
            keyword_collection.insert([[keyword_embedding], [keyword]])
        record = [
            embedding,
            chunk,
            metadata_base["createAt"],
            metadata_base["scrapUrl"],
            metadata_base["urlTitle"],
            metadata_base["dataType"],
            metadata_base["fileName"],
            metadata_base["fileIndex"],
            ", ".join(keywords)
        ]
        insert_data.append(record)

    # Transpose list of records (list of rows) to list of columns (fields)
    fields = list(map(list, zip(*insert_data)))
    return fields

# MySQL에서 전체 데이터 처리 후 Milvus 삽입
def embed_all_json_from_mysql():
    raw_data = load_data_from_mysql()
    total_chunks = 0

    logging.info("Starting embedding process for all MySQL data...")

    for item in raw_data:
        partition_name = item.get("category", "default").replace("-", "_")

        logging.info(f"Processing item ID: {item.get('id')} with category: {partition_name}")

        if not main_collection.has_partition(partition_name):
            main_collection.create_partition(partition_name)

        records, part = process_json_item(item), partition_name
        total_chunks += len(records[0])
        main_collection.insert(records, partition_name=part)

        logging.info(f"Inserted {len(records[0])} chunks into partition '{partition_name}'")

    main_collection.flush()
    logging.info(f"Total {total_chunks} chunks inserted into Milvus.")


# 실행
if __name__ == "__main__":
    embed_all_json_from_mysql()
