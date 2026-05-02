from sklearn.feature_extraction.text import TfidfVectorizer

def create_vectorizer():
    return TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        min_df=5,
        ngram_range=(1,2)   # BIGRAMS = more powerful
    )