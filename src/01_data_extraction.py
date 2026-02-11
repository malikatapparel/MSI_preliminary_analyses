"""
Script: 01_data_extraction.py
Project: Motivational Salience Index (MSI)
Authors: Marie Pittet, adapted by Malika Tapparel
Description: Adapted scripts from Marie Pittet's project to fit Stimulus-Response Compatibility task
Turns event-level task logs into:
1) trial_df: one row per trial
2) item_df: one row per food item per task 
- Merges VAS liking score  for each food item 
"""
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------
df_raw = pd.read_csv("../data/raw/SRC_Long_full.csv") # raw event-level log with one row per event (trials kept 200 ms< RT < 2000 ms)
df_vas = pd.read_csv("../data/raw/liking_imageID.csv", sep = ";")

# ------------------------------------------------------------
# 2) Keep relevant columns for df_raw
# ------------------------------------------------------------
columns = ["imageUID",
           "ParticipantID",
           "session",
           "trial_id",
           "Instruction",
           "FirstPressOutcome",
           "FirstPressRT",  
           ]

df_raw = df_raw[columns].copy()
df_raw = df_raw[df_raw['session']=='pre'].copy() # only keep pre session for now
# Rename columns
df_raw = df_raw.rename(columns={
    "imageUID": "item_id",
    "ParticipantID": "sbj",
    "trial_id": "trial_index",
    "FirstPressOutcome": "outcome",
    "FirstPressRT": "rt",
    "Instruction": "trial_type"
})

# ------------------------------------------------------------
# 3) VAS liking: clean + merge + drop missing
# ------------------------------------------------------------
df_vas = df_vas.rename(columns={"imageID": "item_id", "score": "vas_score"})

df_vas["sbj"] = df_vas["sbj"].map(lambda x: x[2:]) # Remove "PE" prefix from subject IDs
df_vas["sbj"] = pd.to_numeric(df_vas["sbj"], errors="coerce").astype("Int64")

df_vas["item_id"]      = pd.to_numeric(df_vas["item_id"], errors="coerce").astype("Int64")
df_vas["vas_score"]    = pd.to_numeric(df_vas["vas_score"], errors="coerce")

# if the same item appears more than once, average it
df_vas = (
    df_vas.groupby(["sbj", "item_id"], as_index=False)["vas_score"]
          .mean()
)

#
# ------------------------------------------------------------
# 4) Trial-level summary: keep only trials with VAS + add RT for towards vs away
# ------------------------------------------------------------

# merge onto trial_df and drop trials without VAS
trial_df = df_raw.merge(df_vas, on=["sbj", "item_id"], how="left")
trial_df = trial_df.dropna(subset=["vas_score"]).copy()

# 
# ------------------------------------------------------------
# 5) Item-level summary: separate Towards vs away + RT only on responses
# ------------------------------------------------------------

# Basic outcome flags
trial_df["is_correct_towards"]  = (trial_df["outcome"] == "Correct") & (trial_df["trial_type"] == "towards")
trial_df["is_correct_away"]  = (trial_df["outcome"] == "Correct") & (trial_df["trial_type"] == "away")
trial_df["is_miss_towards"] = (trial_df["outcome"] == "Missed") & (trial_df["trial_type"] == "towards")
trial_df["is_miss_away"] = (trial_df["outcome"] == "Missed") & (trial_df["trial_type"] == "away")

# Get RT for trials towards and away
trial_df['rt_correct_towards'] = trial_df['rt'].where((trial_df['trial_type'] == 'towards') & (trial_df["is_correct_towards"] == True), np.nan)
trial_df['rt_correct_away'] = trial_df['rt'].where((trial_df['trial_type'] == 'away') & (trial_df["is_correct_away"] == True), np.nan)

trial_df['rt_miss_towards'] = trial_df['rt'].where((trial_df['trial_type'] == 'towards') & (trial_df["is_miss_towards"] == True), np.nan)
trial_df['rt_miss_away'] = trial_df['rt'].where((trial_df['trial_type'] == 'away') & (trial_df["is_miss_away"] == True), np.nan)   
# Group key
g = ["sbj", "item_id"]


# Aggregate counts + RT summaries
item_df = (
    trial_df.dropna(subset=["item_id"])
    .groupby(g, as_index=False)
    .agg(
        vas_score=("vas_score", "first"),
        n_trials=("trial_index", "count"),

        n_correct_towards=(("is_correct_towards"), "sum"),
        n_correct_away=(("is_correct_away"), "sum"),
        n_miss_towards=(("is_miss_towards"), "sum"),
        n_miss_away=(("is_miss_away"), "sum"),


        # RT summaries for correct trials
        mean_rt_correct_towards=("rt_correct_towards", "mean"),
        median_rt_correct_towards=("rt_correct_towards", "median"),
        mean_rt_correct_away=("rt_correct_away", "mean"),
        median_rt_correct_away=("rt_correct_away", "median"),

        # RT summaries for missed trials
        mean_rt_miss_towards=("rt_miss_towards", "mean"),
        median_rt_miss_towards=("rt_miss_towards", "median"),
        mean_rt_miss_away=("rt_miss_away", "mean"),
        median_rt_miss_away=("rt_miss_away", "median"),
    )
)

# Compute go/nogo accuracies from counts
towards_den = item_df["n_correct_towards"] + item_df["n_miss_towards"]
away_den = item_df["n_correct_away"] + item_df["n_miss_away"]

item_df["acc_towards"] = item_df["n_correct_towards"] / towards_den
item_df.loc[towards_den == 0, "acc_towards"] = np.nan

item_df["acc_away"] = item_df["n_correct_away"] / away_den
item_df.loc[away_den == 0, "acc_away"] = np.nan

# ------------------------------------------------------------
# 8) Extracting the dataframes for later use
# ------------------------------------------------------------
trial_df.to_csv("../data/extracted/trial_df.csv", index=False)
item_df.to_csv("../data/extracted/item_df.csv", index=False)

