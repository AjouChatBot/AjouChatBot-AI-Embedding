from datetime import datetime
import pymysql
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from openai import OpenAI
from keyword_extractor import extract_keywords
import os
import logging
import re
import time
from contextlib import contextmanager

# 로깅 설정 강화
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()

# Milvus 연결
connections.connect(
    host=os.getenv("MILVUS_HOST"),
    port=os.getenv("MILVUS_PORT")
)
logger.info("Milvus 연결 완료")

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

@contextmanager
def get_mysql_connection():
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DATABASE"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    try:
        yield conn
    finally:
        conn.close()

def acquire_work_item(retry_interval=30):
    """작업할 항목을 가져오고 잠금을 설정합니다.
    
    Args:
        retry_interval (int): 재시도 간격(초)
    """
    while True:
        with get_mysql_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    # 트랜잭션 시작
                    conn.begin()
                    
                    # 처리할 항목 선택 및 잠금
                    sql = """
                        SELECT c.id, content_text AS text, c.created_at AS createAt,
                               scrap_url AS scrapUrl, url_title AS urlTitle, c.data_type AS dataType, 
                               org_file_name AS fileName, file_ref_index AS fileIndex,
                               c.category AS category
                        FROM contents c
                        LEFT JOIN scrap_info si ON c.log_id = si.id
                        WHERE content_text IS NOT NULL 
                        AND TRIM(content_text) != ''
                        AND is_embedded = 0
                        AND is_processing = 0
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    """
                    cursor.execute(sql)
                    result = cursor.fetchone()
                    
                    if result:
                        # 항목을 처리 중으로 표시
                        update_sql = """
                            UPDATE contents 
                            SET is_processing = 1
                            WHERE id = %s
                        """
                        cursor.execute(update_sql, (result['id'],))
                        conn.commit()
                        
                        data_type_map = {0: "text", 1: "img", 2: "document"}
                        if "createAt" in result:
                            result["createAt"] = result["createAt"].isoformat() if isinstance(result["createAt"], (datetime,)) else result["createAt"]
                        if "dataType" in result:
                            result["dataType"] = data_type_map.get(result["dataType"], str(result["dataType"]))
                        
                        logger.info(f"작업 항목 획득 완료: ID {result['id']}")
                        return result
                    else:
                        conn.commit()
                        logger.info(f"처리할 새로운 데이터가 없습니다. {retry_interval}초 후 재시도합니다.")
                        time.sleep(retry_interval)
                        continue
                        
                except Exception as e:
                    conn.rollback()
                    logger.error(f"작업 항목 획득 중 오류 발생: {str(e)}")
                    raise

def mark_as_embedded(content_id, success=True):
    """작업 완료 상태를 업데이트합니다."""
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            try:
                conn.begin()
                if success:
                    update_sql = """
                        UPDATE contents 
                        SET is_embedded = 1,
                            is_processing = 0
                        WHERE id = %s
                    """
                else:
                    update_sql = """
                        UPDATE contents 
                        SET is_processing = 0
                        WHERE id = %s
                    """
                cursor.execute(update_sql, (content_id,))
                conn.commit()
                logger.info(f"작업 상태 업데이트 완료: ID {content_id}, 성공: {success}")
            except Exception as e:
                conn.rollback()
                logger.error(f"작업 상태 업데이트 중 오류 발생: {str(e)}")
                raise

def process_single_item(retry_interval=30):
    """단일 항목을 처리하는 함수"""
    try:
        raw_data = acquire_work_item(retry_interval)
        if not raw_data:
            return False

        category = raw_data.get("category") or "default"
        partition_name = category.replace("-", "_")

        logger.info(f"항목 처리 시작 - ID: {raw_data.get('id')}, 카테고리: {partition_name}")

        if not main_collection.has_partition(partition_name):
            main_collection.create_partition(partition_name)
            logger.info(f"새로운 파티션 생성: {partition_name}")

        records, part = process_json_item(raw_data), partition_name
        total_chunks = len(records[0])
        logger.info(f"청크 생성 완료: {total_chunks}개")

        main_collection.insert(records, partition_name=part)
        logger.info(f"Milvus에 데이터 삽입 완료: {total_chunks}개 청크")

        mark_as_embedded(raw_data.get("id"), True)
        logger.info(f"MySQL 업데이트 완료: ID {raw_data.get('id')}")

        main_collection.flush()
        logger.info("Milvus flush 완료")
        
        return True

    except Exception as e:
        logger.error(f"처리 중 오류 발생 - ID: {raw_data.get('id') if raw_data else 'Unknown'}, 오류: {str(e)}")
        if raw_data:
            mark_as_embedded(raw_data.get("id"), False)
        return False

def embed_all_json_from_mysql(retry_interval=30):
    """모든 데이터를 처리하는 메인 함수"""
    logger.info("임베딩 프로세스 시작")
    
    while True:
        if not process_single_item(retry_interval):
            logger.info("더 이상 처리할 데이터가 없거나 오류가 발생했습니다.")
            break
        time.sleep(1)
    
    logger.info("임베딩 프로세스 종료")

def simple_sentence_split(text):
    sentence_endings = re.compile(r'(?<!\b\d)(?<=[.!?])\s+(?=[A-Z가-힣])')
    sentences = sentence_endings.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]

# 텍스트 청킹 및 임베딩
def process_json_item(item: dict):
    text = item.get("text", "")
    sentences = simple_sentence_split(text)

    chunks = []
    current_chunk = ""
    chunk_size = 500
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

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
        full_chunk = f"{item.get('urlTitle', '')}\n{chunk}"
        embedding = openai.embeddings.create(input=full_chunk, model="text-embedding-ada-002").data[0].embedding
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

# 실행
if __name__ == "__main__":
    # 환경 변수에서 재시도 간격을 가져오거나 기본값 사용
    retry_interval = int(os.getenv("RETRY_INTERVAL", "30"))
    embed_all_json_from_mysql(retry_interval)
