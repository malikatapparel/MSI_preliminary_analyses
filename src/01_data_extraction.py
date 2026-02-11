"""
Script: 01_data_extraction.py
Project: Motivational Salience Index (MSI)
Authors: Marie Pittet, adapted by Malika Tapparel
Description: Adapted scripts from Marie Pittet's project to fit Stimulus-Response Compatibility task
Turns event-level task logs into:
1) trial_df: one row per trial
2) item_df: one row per food item per task 
This 'test' branch will include only: is_correct, is_miss, rt_towards, rt_away
- Merges VAS liking score  for each food item 
"""
# %%
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
# 5) Item-level summary: separate Towards vs away + RT only on correct responses
# ------------------------------------------------------------

# Basic outcome flags
trial_df["is_correct"]  = (trial_df["outcome"] == "Correct")
trial_df["is_miss"] = (trial_df["outcome"] == "Missed")

# Get RT for trials towards and away (only for correct responses, else NaN)
trial_df['rt_towards'] = trial_df['rt'].where((trial_df['trial_type'] == 'towards') & (trial_df["is_correct"] == True), np.nan)
trial_df['rt_away'] = trial_df['rt'].where((trial_df['trial_type'] == 'away') & (trial_df["is_correct"] == True), np.nan)

trial_df['is_correct_towards'] = np.where(
    trial_df['trial_type'] == 'towards', 
    trial_df['is_correct'],  # Keep True/False for towards trials
    np.nan                   # NaN for away trials
)

trial_df['is_correct_away'] = np.where(
    trial_df['trial_type'] == 'away', 
    trial_df['is_correct'],   # Keep True/False for away trials  
    np.nan                   # NaN for towards trials
)

# Group key
g = ["sbj", "item_id"]

# Aggregate counts + RT summaries
item_df = (
    trial_df.dropna(subset=["item_id"])
    .groupby(g, as_index=False)
    .agg(
        vas_score=("vas_score", "first"),
        n_trials=("trial_index", "count"),

        n_correct=(("is_correct"), "sum"),
        n_miss=(("is_miss"), "sum"),


        # RT summaries for correct trials
        mean_rt_towards=("rt_towards", "mean"),
        median_rt_towards=("rt_towards", "median"),
        mean_rt_away=("rt_away", "mean"),
        median_rt_away=("rt_away", "median"),

        # Accuracy
        acc_all = (("is_correct"), "mean"),
        acc_towards=(("is_correct_towards"), "mean"),
        acc_away=(("is_correct_away"), "mean"),
    )
)
# ------------------------------------------------------------
# 8) Compute implicit wanting on mean and median at item level
# ------------------------------------------------------------
item_df['iw_mean'] = item_df['mean_rt_away'] - item_df['mean_rt_towards']
item_df['iw_median'] = item_df['median_rt_away'] - item_df['median_rt_towards']
# ------------------------------------------------------------
# 8) Extracting the dataframes for later use
# ------------------------------------------------------------
trial_df.to_csv("../data/extracted/trial_df.csv", index=False)
item_df.to_csv("../data/extracted/item_df.csv", index=False)

# %%
