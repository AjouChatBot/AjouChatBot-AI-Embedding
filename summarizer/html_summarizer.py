import os
import pymysql
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

def get_db_connection():
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST"),
        port=int(os.getenv("MYSQL_PORT")),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        db=os.getenv("MYSQL_DATABASE"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )

def summarize_with_openai(text, url_title, file_name, file_index):
    prompt = f"""
아래는 웹 페이지의 HTML 원문입니다. 구조를 참고하여 내용을 이해하기 쉬운 서술형 줄글로 바꿔주세요.
페이지 제목은 '{url_title}'이며, 파일 제목은 '{file_name}'입니다. 파일 목차는 '{file_index}'입니다.
모든 내용을 포함하고, 어느 하나도 임의로 빼지 마세요.

원문: {text}
"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def acquire_work_item():
    """작업할 항목을 가져오고 잠금을 설정합니다."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 트랜잭션 시작
            conn.begin()
            
            # 처리할 항목 선택 및 잠금
            sql = """
                SELECT c.id, data, url_title, org_file_name AS file_name, file_ref_index AS file_index
                FROM contents c
                LEFT JOIN scrap_info si ON c.log_id = si.id
                WHERE (c.content_text IS NULL OR c.content_text = '') 
                AND c.data_type = 0
                AND c.is_processing = 0
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
                logger.info(f"작업 항목 획득 완료: ID {result['id']}")
                return result
            else:
                conn.commit()
                return None
                
    except Exception as e:
        conn.rollback()
        logger.error(f"작업 항목 획득 중 오류 발생: {str(e)}")
        raise
    finally:
        conn.close()

def update_text_in_mysql(id, summary):
    """작업 완료 상태를 업데이트합니다."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                UPDATE contents 
                SET content_text = %s,
                    is_processing = 0
                WHERE id = %s
            """
            cursor.execute(sql, (summary, id))
        conn.commit()
        logger.info(f"작업 완료 처리: ID {id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"작업 완료 처리 중 오류 발생: {str(e)}")
        raise
    finally:
        conn.close()

def main():
    retry_interval = 30  # 재시도 간격(초)
    
    while True:
        try:
            # 작업할 항목 가져오기
            item = acquire_work_item()
            
            if item:
                # 항목 처리
                logger.info(f"항목 처리 시작: ID {item['id']}")
                summary = summarize_with_openai(
                    item['data'],
                    item['url_title'],
                    item['file_name'],
                    item['file_index']
                )
                update_text_in_mysql(item['id'], summary)
                logger.info(f"항목 처리 완료: ID {item['id']}")
            else:
                logger.info(f"처리할 새로운 데이터가 없습니다. {retry_interval}초 후 재시도합니다.")
                time.sleep(retry_interval)
                
        except Exception as e:
            logger.error(f"처리 중 오류 발생: {str(e)}")
            time.sleep(retry_interval)

if __name__ == "__main__":
    main()