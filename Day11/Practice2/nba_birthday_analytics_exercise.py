"""
Title: NBA Player Biographical & Birthday Paradox Analysis 
Description: A modular Python exercise script for loading, sanitizing, and analyzing 
             NBA player birth dates to explore shared birthdays and player lookups.

"""

# Step 0: Import necessary libraries
import numpy as np
import pandas as pd
import datetime
# TODO: Import pandas, datetime, and any required modules


def phase_1_discovery_and_sanitization():
    """
    [Phase 1] Data Discovery, Column Standardization & Sanitization (5 Marks)
    - Goal: Load the NBA player CSV file.
    - Standardize column names (remove whitespace, lowercase, snake_case).
    - Sanitize 'Birth Date' into standard datetime format.
    """
    print("\n[PHASE 1] Discovery & Data Sanitization (5 Marks)")
    print("-" * 65)
    # TODO: Load CSV file using pandas (handling potential quote formatting)
    df = pd.read_csv("./player_data.csv")
    # TODO: Standardize column headers (e.g., "Birth Date" -> "birth_date")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    print(df.head())
    # TODO: Parse 'birth_date' column into datetime objects using pd.to_datetime()
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    # print(df["birth_date"].dtype)
    # TODO: Inspect dataset shape and check for missing birth dates
    print("The shape of the dataset is: ")
    print(df.shape)
    print("The null values in the dataset are")
    print(df.isnull().sum())
    # print(df)
    # df["wt"] = df["wt"].bfill()
    # df["birth_date"] = df["birth_date"].ffill()
    # college_mode = df["colleges"].mode().iloc[0]
    # df["colleges"] = df["colleges"].fillna(college_mode)
    # print(college_mode)
    print(df.isnull().sum())
    return df


def phase_2_birthday_paradox_analysis(df):
    """
    [Phase 2] Birthday Paradox & Distribution Analysis (10 Marks)
    - Goal: Analyze the distribution of birth dates across months and days.
    - Calculate and demonstrate the probability of shared birthdays (Birthday Paradox) 
      within subsets of NBA players.
    """
    print("\n[PHASE 2] Birthday Paradox & Date Distribution (10 Marks)")
    print("-" * 65)
    # print(df.head())
    df_clean = df.copy()
    # TODO: Extract month and day components from 'birth_date'
        
    df_clean["month"] = df_clean["birth_date"].dt.month
    df_clean["day"] = df_clean["birth_date"].dt.day
    # print(df_clean.head())
        
    # TODO: Count frequency of birthdays per calendar day
    frequency = df_clean["day"].value_counts().sort_index()
    print("Frequency of birthdays per day:")
    # print(frequency)
    # TODO: Identify dates shared by multiple NBA players in the dataset
    Counts = df_clean.groupby(["month", "day"]).size().reset_index(name="count")
    print(Counts)
    shared_birthdays = Counts[Counts["count"] > 1]
    # BREAKDOWN ofabove query:
    # Counts = df_clean.groupby(["month", "day"]).size()
    # print("Counts: ", Counts.head())
    
    print("Shared birthdays (more than one player):\n", shared_birthdays.head())
    return df_clean


def phase_3_player_lookup_system(df, player_name):
    """
    [Phase 3] Player Detail Retrieval (10 Marks)
    - Goal: Search for a specific player by name (case-insensitive or partial match).
    - Retrieve their biographical details (Position, College, Birth Date).
    """
    print(f"\n[PHASE 3] Player Lookup: Searching for '{player_name}' (10 Marks)")
    print("-" * 65)
    # TODO: Filter dataframe for the specified player name
    # print(df.columns.values)
    player_df = df[df["player"].str.contains(player_name, case=False, na=False)]
    if player_df.empty: 
        print(f"No player found with the name '{player_name}'.")
        return None 
    
    # TODO: Extract and return player details (Position, Ht, Wt, Colleges, Birth Date)
    # print(f"Player details for '{player_name}':")
    # print(player_df[["pos", "ht", "wt", "colleges", "birth_date"]])
    return df


def phase_4_shared_birthday_cohort(df, player_name):
    """
    [Phase 4] Shared Birthday Peer Finder (10 Marks)
    - Goal: Identify the target player's exact birth date.
    - List all other NBA players who share the exact same date of birth.
    """
    print(f"\n[PHASE 4] Shared Birthday Cohort for '{player_name}' (10 Marks)")
    print("-" * 65)
    # TODO: Retrieve the birth date of the target player from Phase 3
    player_df = df[df["player"].str.contains(player_name, case=False, na=False)]
    if player_df.empty:
        print(f"No player found with the name '{player_name}'.")
        return None
    exact_birth_date = player_df["birth_date"].iloc[0]
    # print("exact_birth_date: ",exact_birth_date)
    print(f"{player_name}'s birth date is: {exact_birth_date.date()}")
    # TODO: Query the dataset for all other players born on that exact month and day (or year)
    
    
    excat_player_month = exact_birth_date.month
    excat_player_day = exact_birth_date.day
    player_same = df[(df["birth_date"].dt.month == excat_player_month) & (df["birth_date"].dt.day == excat_player_day)]
    print("Player having same things are: \n", player_same)
    print(player_same[["player", "birth_date"]])
    # # TODO: Output the list of peer players sharing the birthday
    if not player_same.empty:
        print("\nPlayers sharing the same birthday:")
        print(player_same["player"].to_list())
    else:
        print("No player found in this dataset...")
    return df


def phase_5_evaluation_and_summary(df):
    """
    [Phase 5] Evaluation & Reporting (5 Marks)
    - Goal: Summarize statistical findings on player longevity, birth months, 
      and birthday coincidences.
    """
    print("\n[PHASE 5] Evaluation & Summary Reporting (5 Marks)")
    print("-" * 65)
    # TODO: Print final summary statistics regarding dataset coverage and duplicate birthdays
    # df_clean = df.copy()
    summary = df.describe().T
    print(summary)
    Counts = df.groupby(["month", "day"]).size().reset_index(name="count")
    shared_birthdays = Counts[Counts["count"] > 1]
    print(shared_birthdays)
    
    print("Total players involved in shared birthdays:", shared_birthdays["count"].sum())
    
    
    print("Shared birthdays are: \n", shared_birthdays)
    pass


# ==========================================
# THE FACADE FUNCTION (Orchestrator)
# ==========================================

def run_analytics_lifecycle():
    """
    Facade function coordinating the NBA analytics and birthday lookup workflow.
    """
    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE (Out of 40 Marks)")
    print("=" * 65)
    
    # Target player to test lookup and shared birthdays
    target_player = "Kareem Abdul-Jabbar"
    
    # Executing phases chronologically
    df = phase_1_discovery_and_sanitization()
    df = phase_2_birthday_paradox_analysis(df)
    df = phase_3_player_lookup_system(df, target_player)
    df = phase_4_shared_birthday_cohort(df, target_player)
    phase_5_evaluation_and_summary(df)


# ==========================================
# MAIN GUARD
# ==========================================
if __name__ == "__main__":
    run_analytics_lifecycle()