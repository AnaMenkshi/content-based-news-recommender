
import csv
import random
import pandas as pd
from datetime import datetime, timedelta


# Load users & news
users = pd.read_csv("003-users-updated.csv")
news = pd.read_csv("001-news-items.csv")  # expects at least 'id' and 'content'


# Basic validation
if "user_id" not in users.columns:
    raise ValueError("Missing required column 'user_id' in users dataset")

if "id" not in news.columns:
    raise ValueError("Missing required column 'id' in news dataset")


# Split location into city and country
if "location" in users.columns:
    loc = users["location"].astype(str).str.split(",", n=1, expand=True)
    users["city"] = loc[0].fillna("").astype(str).str.strip()
    users["country"] = (loc[1] if loc.shape[1] > 1 else "").fillna("").astype(str).str.strip()
else:
    raise ValueError("No 'location' column found in users dataset")


# Configuration
OUTPUT_FILE = "004-user-behavior-fixed.csv"
INTERACTIONS = [("view", 0.3), ("click", 0.6), ("like", 0.9)]  # base strength
WEIGHTS = [0.6, 0.3, 0.1]  # probability of each interaction type
MIN_READS = 5
MAX_READS = 20
START_DATE = datetime(2025, 1, 1, 8, 0, 0)


# Helper functions
def weighted_interaction():
    """Pick an interaction type with weighted probability"""
    return random.choices(INTERACTIONS, weights=WEIGHTS, k=1)[0]

def relevance_score(user_city, user_country, news_content, base_strength):

    """Boost interaction strength if news mentions user's city or country (with realistic safeguards)."""

    content_lower = str(news_content).lower()

    city = "" if pd.isna(user_city) else str(user_city).strip().lower()
    country = "" if pd.isna(user_country) else str(user_country).strip().lower()

    boost = 0.0

    # IMPORTANT: avoid boosting when city/country is empty ("" is always "in" any string)
    if city and city in content_lower:
        boost += 0.2
    if country and country in content_lower:
        boost += 0.2

    # Small noise so strengths aren’t all identical per interaction type
    noise = random.uniform(-0.05, 0.05)

    score = base_strength + boost + noise
    return max(0.0, min(score, 1.0))  # clip to [0, 1]


# Generate dataset
rows = []

for _, user in users.iterrows():
    user_id = user.get("user_id", None)
    if pd.isna(user_id):
        continue

    city = user.get("city", "")
    country = user.get("country", "")

    # Each user reads between MIN_READS and MAX_READS news items
    num_reads = random.randint(MIN_READS, MAX_READS)

    # If dataset is smaller than requested reads, sample with replacement to keep it "complete"
    replace = num_reads > len(news)
    sampled_news = news.sample(
        n=num_reads,
        replace=replace,
        random_state=random.randint(0, 10000)
    )

    current_time = START_DATE
    for _, item in sampled_news.iterrows():
        news_id = item.get("id", None)
        if pd.isna(news_id):
            continue

        interaction_type, base_strength = weighted_interaction()

        # content may be missing; default to empty string
        content = item.get("content", "")
        score = relevance_score(city, country, content, base_strength)

        # Increment timestamp randomly
        current_time += timedelta(minutes=random.randint(1, 15))

        rows.append([
            str(user_id),
            str(news_id),
            interaction_type,
            round(score, 6),
            current_time.isoformat()
        ])


# Write CSV
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "user_id",
        "news_id",
        "interaction_type",
        "interaction_strength",
        "timestamp"
    ])
    writer.writerows(rows)

print(f" Generated {len(rows)} rows in {OUTPUT_FILE}")
