import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer


# Parameters
TIME_DECAY_LAMBDA = 0.01
SMOOTH_FACTOR = 0.3
POPULARITY_ALPHA = 0.3  # weight of popularity penalty
RANDOM_STATE = 42


# Step 1: Load datasets

users = pd.read_csv("003-users-updated.csv")
news = pd.read_csv("001-news-items.csv")
behavior = pd.read_csv("004-user-behavior-fixed.csv")

news = news.rename(columns={"id": "news_id"})


# Step 2: Location cleanup
users[['city', 'country']] = users['location'].str.split(',', n=1, expand=True)
users['city'] = users['city'].str.strip()
users['country'] = users['country'].str.strip()


# Step 3: Parse timestamps
behavior["timestamp"] = pd.to_datetime(behavior["timestamp"], errors="coerce")
news["pubDate"] = pd.to_datetime(news["pubDate"], errors="coerce")

latest_time = behavior["timestamp"].max()


# Step 4: Time-aware train/test split per user
train_list, test_list = [], []

for uid, group in behavior.groupby("user_id"):
    group = group.sort_values("timestamp")
    if len(group) <= 2:
        train_list.append(group)
    else:
        split_idx = int(len(group) * 0.8)
        train_list.append(group.iloc[:split_idx])
        test_list.append(group.iloc[split_idx:])

train_behavior = pd.concat(train_list).reset_index(drop=True)
test_behavior = pd.concat(test_list).reset_index(drop=True)

train_behavior.to_csv("004-user-behavior-train.csv", index=False)
test_behavior.to_csv("004-user-behavior-test.csv", index=False)

print(f" Time-aware split done. Train: {len(train_behavior)}, Test: {len(test_behavior)}")


# Step 5: Load Albanian stopwords
with open("stop-words.json", "r", encoding="utf-8") as f:
    stopwords = json.load(f)


# Step 6: TF-IDF topic vectors for news
news["text"] = news["title"].fillna("") + " " + news["content"].fillna("")

tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words=stopwords,
    ngram_range=(1, 2),
    sublinear_tf=True
)

news_tfidf = tfidf.fit_transform(news["text"])
news_vectors = pd.DataFrame(
    news_tfidf.toarray(),
    index=news["news_id"],
    columns=tfidf.get_feature_names_out()
)


# Step 7: Popularity (training only)
popularity = train_behavior.groupby("news_id")["interaction_strength"].sum()
popularity = np.log1p(popularity)
popularity = popularity / popularity.max()
popularity = popularity.reindex(news_vectors.index).fillna(0)


# Step 8: Build IMPROVED user profiles (TIME DECAY + POPULARITY + mean pooling)
user_profiles = {}

for uid, group in train_behavior.groupby("user_id"):
    read_news = group["news_id"].values
    vectors = news_vectors.loc[read_news].values

    # Interaction strength
    strength = group["interaction_strength"].values

    # Time decay
    age_days = (latest_time - group["timestamp"]).dt.days.fillna(0).values
    time_weight = np.exp(-TIME_DECAY_LAMBDA * age_days)

    # Popularity penalty (less popular news slightly upweighted)
    pop_weight = 1 - POPULARITY_ALPHA * popularity.loc[read_news].values

    # Final weights
    weights = (strength * time_weight * pop_weight + SMOOTH_FACTOR).reshape(-1, 1)

    # Weighted MEAN pooling
    profile = np.sum(vectors * weights, axis=0) / np.sum(weights)

    # Normalize
    profile /= (np.linalg.norm(profile) + 1e-8)

    user_profiles[uid] = profile


# Step 9: Convert profiles dict to DataFrame
user_profiles_df = pd.DataFrame.from_dict(
    user_profiles,
    orient="index",
    columns=news_vectors.columns
)
user_profiles_df.index.name = "user_id"


# Step 10: Merge user metadata
user_profiles_final = user_profiles_df.reset_index().merge(
    users[['user_id', 'username', 'city', 'country']],
    on='user_id',
    how='left'
)


# Step 11: Save 005-user-profiles.csv
user_profiles_final.to_csv("005-user-profiles.csv", index=False)

print(" 005-user-profiles.csv created (TIME DECAY + POPULARITY + mean pooling)")
print(user_profiles_final.head())
