from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import load_dataset, validate_dataset
from src.preprocessing import clean_text
from src.train import prepare_dataframe


TRAIN_PATH = Path("data/processed/train_split.xlsx")


@lru_cache(maxsize=1)
def load_retrieval_index():
    df = load_dataset(str(TRAIN_PATH))
    validate_dataset(df)
    df = prepare_dataframe(df).copy()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["clean_text"])

    return df, vectorizer, X


def retrieve_examples(query_text: str, k_per_class: int = 3, max_chars: int = 180):
    df, vectorizer, X = load_retrieval_index()

    query_clean = clean_text(query_text)
    query_vec = vectorizer.transform([query_clean])

    similarities = cosine_similarity(query_vec, X)[0]

    temp = df.copy()
    temp["similarity"] = similarities
    temp = temp.sort_values("similarity", ascending=False)

    examples = []

    for label_value in [0, 1]:
        subset = temp[temp["label"] == label_value].head(k_per_class)

        for _, row in subset.iterrows():
            label_text = "anorexia" if int(row["label"]) == 1 else "control"
            raw_text = str(row["tweet_text"]) if "tweet_text" in row else str(row.get("text", ""))
            raw_text = raw_text.replace("\n", " ").strip()

            if len(raw_text) > max_chars:
                raw_text = raw_text[:max_chars].rstrip() + "..."

            examples.append({
                "label": label_text,
                "text": raw_text,
                "similarity": float(row["similarity"])
            })

    return examples