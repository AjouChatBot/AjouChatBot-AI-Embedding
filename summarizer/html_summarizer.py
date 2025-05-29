import os
import pymysql
from openai import OpenAI
from dotenv import load_dotenv

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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

def load_data_from_mysql():
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            sql = """
                SELECT c.id, data, url_title, org_file_name AS file_name, file_ref_index AS file_index
                FROM contents c
                LEFT JOIN scrap_info si ON c.log_id = si.id
                WHERE (c.content_text IS NULL OR c.content_text = '') AND c.data_type = 0  -- 아직 요약 안 된 항목 중 data_type=0
            """
            cursor.execute(sql)
            return cursor.fetchall()
    finally:
        conn.close()

def update_text_in_mysql(id, summary):
    conn = get_db_connection()

    try:
        with conn.cursor() as cursor:
            sql = "UPDATE contents SET content_text = %s WHERE id = %s"
            cursor.execute(sql, (summary, id))
        conn.commit()
    finally:
        conn.close()

def main():
    items = load_data_from_mysql()
    for item in items:
        print(f"Processing ID {item['id']}...")
        plain_text = item['data']  # 태그 포함 원본 HTML 사용
        summary = summarize_with_openai(plain_text, item['url_title'], item['file_name'], item['file_index'])
        update_text_in_mysql(item['id'], summary)

if __name__ == "__main__":
    main()