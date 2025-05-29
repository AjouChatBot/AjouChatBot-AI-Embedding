# Python 3.9 베이스 이미지 사용
FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# numpy를 먼저 설치
RUN pip install --no-cache-dir numpy

# requirements.txt 복사 및 나머지 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "html_summarizer.py"]