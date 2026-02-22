### CONTENT-BASED NEWS RECOMMENDER SYSTEM
#### PROJECT OVERVIEW

This project implements a time-aware, hybrid content-based recommendation system for news articles.

#### The system:

Cleans and standardizes user, news, and interaction data

Represents news using TF-IDF (unigrams + bigrams)

Builds weighted user profiles

Generates personalized recommendations

Evaluates performance using precision@K and recall@K

#### The approach combines:

Content similarity

Interaction strength

Time decay

Popularity prior

to create a balanced and personalized recommendation pipeline.

#### PROJECT FILES

users.py

Cleans and updates user dataset

Generates 003-users-updated.csv (1000 rows × 3 columns)

004.py

Processes user behavior data

Generates 004-user-behaviour-fixed.csv (12466 × 5)

005.py

Builds user profiles

Outputs 005-user-profiles.csv (large feature matrix)

testnew.py

Generates recommendations

Performs evaluation (200,002 × 3 recommendations dataset)

### SYSTEM PIPELINE

#### DATA LOADING AND CLEANING

Datasets:

Users (metadata including location)

News (title, content, timestamps)

User behavior (interactions)

Cleaning steps:

Standardize timestamps

Normalize user locations (city, country)

Ensure consistent user IDs and news IDs

Handle missing or malformed entries

### TIME-AWARE TRAIN / TEST SPLIT

##### Strategy:

Per-user chronological split

First 80% → Training

Last 20% → Test

##### Result:

Train interactions: 9597

Test interactions: 2868

This prevents data leakage and ensures realistic evaluation.

### NEWS REPRESENTATION

Text construction:

Concatenate title + content

#### Vectorization:

TF-IDF

N-grams: 1–2

Maximum features: 8,000 (005.py) / 20,000 (testnew.py)

Albanian stopwords

Sublinear TF scaling

Each news article is converted into a numerical vector.

#### POPULARITY PRIOR

Using training data only:

Sum interaction strengths per news

Apply log scaling

Normalize to range [0, 1]

Popularity is later used as a penalty factor to reduce dominance of highly popular articles.

#### USER PROFILE CONSTRUCTION

For each user:

Retrieve TF-IDF vectors of interacted news

Compute weights:

interaction_strength × time_decay × popularity_penalty + smoothing

Where:

Interaction strength reflects engagement

Time decay gives more importance to recent interactions

Popularity penalty reduces bias toward viral content

Smoothing prevents zero weights

Apply weighted mean pooling

Apply L2 normalization

#### Output:

005-user-profiles.csv
(TIME DECAY + POPULARITY + MEAN POOLING)

HYBRID SCORING

For each user:

Compute cosine similarity between user profile and all news

Combine similarity with popularity prior:

final_score = similarity + (POPULARITY_WEIGHT × popularity)

This produces a hybrid personalized ranking.

### RECOMMENDATION GENERATION

Rank news by final score

Exclude already seen news

Select Top-K recommendations (default K = 200)

Save results to CSV

#### EVALUATION

Metrics used:

Precision@K (macro average)

Recall@K (macro average)

Evaluation is done per user and averaged across all users.

Results (Top-200):

Precision@200: 0.0030
Recall@200: 0.2104

### INTERPRETATION OF RESULTS

The system favors coverage (recall) over exact relevance (precision).

Possible reasons:

Large TOP_K (200) dilutes precision

Mean pooling may oversimplify user preferences

TF-IDF features capture general relevance but not deep personalization

Popularity weighting increases recall but reduces specificity

### STRENGTHS

Time-aware modeling

Hybrid scoring (similarity + popularity)

Personalized user profiles

Proper train/test split

Macro-averaged evaluation

Scalable vectorized implementation

#### LIMITATIONS

TF-IDF lacks semantic depth

Mean pooling oversimplifies preferences

No collaborative filtering

No neural embeddings

Precision remains low

### FUTURE IMPROVEMENTS

Use advanced embeddings

Word2Vec

FastText

BERT-based embeddings

Sentence Transformers

Improve ranking

Learning-to-rank models



#### TECHNOLOGIES USED

Python

Pandas

NumPy

Scikit-learn

TF-IDF

Cosine Similarity



#### LICENSE

MIT License
