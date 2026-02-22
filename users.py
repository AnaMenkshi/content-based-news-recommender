import pandas as pd
import numpy as np


# Step 1: Load the original users dataset
users_df = pd.read_csv("003-users.csv")


# Step 2: Make a copy to update locations
users_copy_df = users_df.copy()

#  Rename column: id -> user_id
users_copy_df.rename(columns={"id": "user_id"}, inplace=True)

# Step 3: Define locations (City, Country)
locations = [
    "Tirana, Albania",
    "Shkodër, Albania",
    "Elbasan, Albania",
    "Fier, Albania",
    "Hague, Netherlands",
    "Lushnje, Albania",
    "Librazhd, Albania",
    "Përrenjas, Albania",
    "Korçë, Albania",
    "Pristina, Kosovo",
    "Washington, United States",
    "Rome, Italy",
    "Berlin, Germany",
    "Paris, France",
    "Moscow, Russia",
    "Athens, Greece",
    "Minsk, Belarus",
    "Kyiv, Ukraine"
]


# Step 4: Randomly assign locations to users
np.random.seed(42)
# for reproducibility
users_copy_df['location'] = np.random.choice(locations, size=len(users_copy_df))


# Step 5: Save the updated copy
users_copy_df.to_csv("003-users-updated.csv", index=False)
print(" Updated users dataset with locations saved as '003-users-updated.csv'")
