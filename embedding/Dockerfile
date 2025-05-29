# Python 3.9 베이스 이미지 사용
FROM --platform=linux/amd64 python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# numpy를 먼저 설치
RUN pip install --no-cache-dir numpy

# 필요한 Python 패키지 설치를 위한 requirements.txt 복사
COPY requirements.txt .

# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 헬스체크를 위한 포트 노출 (선택사항)
EXPOSE 8080

# 애플리케이션 실행
CMD ["python", "Milvus_embedding.py"] 