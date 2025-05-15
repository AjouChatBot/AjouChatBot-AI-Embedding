from kiwipiepy import Kiwi
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)
kiwi = Kiwi()

korean_stopwords = [
    "그", "이", "저", "것", "수", "등", "들", "및", "또는", "그리고",
    "그러나", "하지만", "때문에", "그래서", "하지만", "따라서",
    "에서", "에게", "으로", "부터", "까지", "보다", "처럼", "중에",
    "는", "은", "이", "가", "을", "를", "의", "에", "도", "만", "로",
    "과", "와", "해서", "하여", "하기", "하는", "하였다", "있다",
    "없다", "이다", "되다", "한", "된", "하는", "하게", "같은", "것이다"
]

def extract_keywords(text: str, top_n: int = 10):
    raw_keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words=korean_stopwords,
        top_n=top_n * 3
    )

    # 명사로만 구성된 키워드 필터링
    noun_keywords = []
    for phrase, _ in raw_keywords:
        tokens = kiwi.analyze(phrase)[0][0]
        if all(pos in ['NNG', 'NNP'] for _, pos, _, _ in tokens):
            noun_keywords.append(phrase)

    return noun_keywords[:top_n]

if __name__ == "__main__":
    print("분석할 텍스트를 입력하세요:")
    user_input = input()
    keywords = extract_keywords(user_input, top_n=5)
    print("\n\n추출된 키워드:", keywords)