import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Configuration
TOP_K = 200
SMOOTH_FACTOR = 0.3
POPULARITY_WEIGHT = 0.35
TIME_DECAY_LAMBDA = 0.01

TRAIN_BEHAVIOR_FILE = "004-user-behavior-train.csv"
TEST_BEHAVIOR_FILE = "004-user-behavior-test.csv"
NEWS_FILE = "001-news-items.csv"
STOPWORDS_FILE = "stop-words.json"
OUTPUT_FILE = "recommendations.csv"


# Step 1: Load data
news = pd.read_csv(NEWS_FILE)
train_behavior = pd.read_csv(TRAIN_BEHAVIOR_FILE)
test_behavior = pd.read_csv(TEST_BEHAVIOR_FILE)

news = news.rename(columns={"id": "news_id"})


# Step 2: TF-IDF topic vectors
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    stopwords = json.load(f)

news["text"] = news["title"].fillna("") + " " + news["content"].fillna("")

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    stop_words=stopwords,
    sublinear_tf=True
)

news_tfidf = tfidf.fit_transform(news["text"])
news_vectors = pd.DataFrame(
    news_tfidf.toarray(),
    index=news["news_id"]
)


# Step 3: Popularity prior
popularity = train_behavior.groupby("news_id")["interaction_strength"].sum()
popularity = np.log1p(popularity)
popularity = popularity / popularity.max()
popularity = popularity.reindex(news_vectors.index).fillna(0).values


# Step 4: Build user profiles with time decay + popularity
latest_time = pd.to_datetime(train_behavior["timestamp"]).max()
user_profiles = {}

for uid, group in train_behavior.groupby("user_id"):
    read_news = group["news_id"].values
    vectors = news_vectors.loc[read_news].values

    # Interaction strength
    strength = group["interaction_strength"].values

    # Time decay
    timestamps = pd.to_datetime(group["timestamp"])
    age_days = (latest_time - timestamps).dt.days.values
    time_weight = np.exp(-TIME_DECAY_LAMBDA * age_days)

    # Popularity penalty for profile weighting
    pop_weight = 1 - POPULARITY_WEIGHT * popularity[np.isin(news_vectors.index, read_news)]

    # Final weight per interaction
    weights = (strength * time_weight * pop_weight + SMOOTH_FACTOR).reshape(-1, 1)

    # Weighted MEAN pooling for user profile
    profile = np.sum(vectors * weights, axis=0) / np.sum(weights)
    profile /= (np.linalg.norm(profile) + 1e-8)

    user_profiles[uid] = profile

user_profiles_df = pd.DataFrame.from_dict(
    user_profiles,
    orient="index",
    columns=tfidf.get_feature_names_out()
)


# Step 5: Hybrid scoring
similarity = cosine_similarity(
    user_profiles_df.values,
    news_vectors.values
)

scores = (1 - POPULARITY_WEIGHT) * similarity + POPULARITY_WEIGHT * popularity

user_ids = user_profiles_df.index.values
news_ids = news_vectors.index.values


# Step 6: Save recommendations
recs = []
for i, uid in enumerate(user_ids):
    top_indices = np.argsort(-scores[i])[:TOP_K]
    for j in top_indices:
        recs.append({
            "user_id": uid,
            "news_id": news_ids[j],
            "score": scores[i, j]
        })

pd.DataFrame(recs).to_csv(OUTPUT_FILE, index=False)
print(f"Recommendations saved to {OUTPUT_FILE}")


# Step 7: Evaluation metrics (per-user averaging)
precision_list, recall_list = [], []

for i, uid in enumerate(user_ids):
    seen = set(train_behavior[train_behavior["user_id"] == uid]["news_id"])
    ranked = [nid for nid in news_ids[np.argsort(-scores[i])] if nid not in seen]

    relevant = test_behavior[test_behavior["user_id"] == uid]["news_id"].tolist()
    if not relevant:
        continue

    precision_list.append(len(set(ranked[:TOP_K]) & set(relevant)) / TOP_K)
    recall_list.append(len(set(ranked[:TOP_K]) & set(relevant)) / len(relevant))


# Step 8: Results
print(f"\n Evaluation Results (Top-{TOP_K})")
print(f"Precision@{TOP_K} (macro avg): {np.mean(precision_list):.4f}")
print(f"Recall@{TOP_K} (macro avg):    {np.mean(recall_list):.4f}")
