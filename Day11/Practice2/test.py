# ============================================================
# NBA PLAYER BIOGRAPHICAL & BIRTHDAY PARADOX ANALYSIS
# ============================================================

# STEP 0: IMPORT LIBRARIES

import pandas as pd
import numpy as np
import re
import math
from pathlib import Path
from datetime import datetime, date


# ============================================================
# PHASE 1: DISCOVERY AND SANITIZATION
# ============================================================

def phase_1_discovery_and_sanitization():

    print("\n[PHASE 1] DISCOVERY & DATA SANITIZATION")
    print("-" * 65)

    # Load CSV
    file_path = Path("nba_players.csv")

    if not file_path.exists():
        raise FileNotFoundError(
            "nba_players.csv not found! Place the CSV in the same folder."
        )

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    print("\nDataset Shape:", df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Records:")
    print(df.head())

    # Handle alternative column names
    column_mapping = {
        "name": "player",
        "player_name": "player",
        "birthdate": "birth_date",
        "birth": "birth_date",
        "ht": "height",
        "wt": "weight",
        "colleges": "college"
    }

    df.rename(columns=column_mapping, inplace=True)

    # Validate required columns
    if "birth_date" not in df.columns:
        raise ValueError("Birth Date column is missing!")

    if "player" not in df.columns:
        raise ValueError("Player/Name column is missing!")

    # Convert birth_date into datetime
    df["birth_date"] = pd.to_datetime(
        df["birth_date"],
        errors="coerce"
    )

    # Remove invalid or missing birth dates
    print("\nMissing Birth Dates Before Cleaning:",
          df["birth_date"].isna().sum())

    df = df.dropna(subset=["birth_date"]).copy()

    # Remove rows without player names
    df = df.dropna(subset=["player"]).copy()

    # Extract date components
    df["birth_year"] = df["birth_date"].dt.year
    df["birth_month"] = df["birth_date"].dt.month
    df["birth_day"] = df["birth_date"].dt.day

    # Create month-day combination (ignores birth year)
    df["birthday"] = df["birth_date"].dt.strftime("%m-%d")

    print("\nCleaned Dataset Shape:", df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    print("\nSanitization Completed!")

    return df


# ============================================================
# PHASE 2: BIRTHDAY PARADOX AND DISTRIBUTION ANALYSIS
# ============================================================

def phase_2_birthday_paradox_analysis(df):

    print("\n[PHASE 2] BIRTHDAY PARADOX ANALYSIS")
    print("-" * 65)

    # Extract month and day
    df = df.copy()

    df["birth_month"] = df["birth_date"].dt.month
    df["birth_day"] = df["birth_date"].dt.day

    # --------------------------------------------------------
    # 1. Birth distribution by month
    # --------------------------------------------------------

    print("\nBIRTHDAYS BY MONTH:")

    month_distribution = (
        df["birth_month"]
        .value_counts()
        .sort_index()
    )

    print(month_distribution)

    # --------------------------------------------------------
    # 2. Count players born on each calendar day
    # --------------------------------------------------------

    print("\nBIRTHDAY FREQUENCY:")

    birthday_frequency = (
        df.groupby("birthday")["player"]
        .count()
        .sort_values(ascending=False)
    )

    print(birthday_frequency.head(15))

    # --------------------------------------------------------
    # 3. Find shared birthdays
    # --------------------------------------------------------

    shared_birthdays = birthday_frequency[
        birthday_frequency > 1
    ]

    print("\nSHARED BIRTHDAYS:")

    print(shared_birthdays)

    # --------------------------------------------------------
    # 4. Birthday Paradox theoretical probability
    # --------------------------------------------------------

    def birthday_probability(n):

        if n > 365:
            return 1.0

        if n < 2:
            return 0.0

        probability_no_match = 1.0

        for i in range(n):
            probability_no_match *= (365 - i) / 365

        return 1 - probability_no_match

    print("\nBIRTHDAY PARADOX PROBABILITY:")

    for n in [10, 20, 23, 30, 40, 50]:

        probability = birthday_probability(n)

        print(
            f"{n} people: {probability * 100:.2f}% "
            "chance of shared birthday"
        )

    # --------------------------------------------------------
    # 5. Actual dataset probability
    # --------------------------------------------------------

    total_players = len(df)

    players_with_shared_birthday = df[
        df.duplicated(subset=["birthday"], keep=False)
    ]

    if total_players > 0:

        actual_percentage = (
            len(players_with_shared_birthday) / total_players
        ) * 100

    else:
        actual_percentage = 0

    print("\nACTUAL DATASET STATISTICS:")

    print("Total Players:", total_players)

    print("Players with Shared Birthdays:",
          len(players_with_shared_birthday))

    print("Percentage of Players Sharing Birthdays:",
          round(actual_percentage, 2), "%")

    return df


# ============================================================
# PHASE 3: PLAYER LOOKUP SYSTEM
# ============================================================

def phase_3_player_lookup_system(df, player_name):

    print(f"\n[PHASE 3] PLAYER LOOKUP: {player_name}")
    print("-" * 65)

    # Case-insensitive partial matching
    matches = df[
        df["player"].astype(str).str.contains(
            re.escape(player_name),
            case=False,
            na=False
        )
    ]

    if matches.empty:

        print("Player not found!")

        return None

    print("\nMatching Players:")
    print(matches["player"].tolist())

    # Prefer exact match if available
    exact_match = matches[
        matches["player"].str.strip().str.lower()
        == player_name.strip().lower()
    ]

    player = (
        exact_match.iloc[0]
        if not exact_match.empty
        else matches.iloc[0]
    )

    # Retrieve available biographical columns
    required_details = [
        "player",
        "position",
        "height",
        "weight",
        "college",
        "birth_date"
    ]

    available_details = [
        col for col in required_details
        if col in df.columns
    ]

    player_details = player[available_details]

    print("\nPLAYER BIOGRAPHICAL DETAILS:")

    print(player_details)

    return player


# ============================================================
# PHASE 4: SHARED BIRTHDAY COHORT
# ============================================================

def phase_4_shared_birthday_cohort(df, player_name):

    print(f"\n[PHASE 4] SHARED BIRTHDAY COHORT: {player_name}")
    print("-" * 65)

    # Find target player
    player = phase_3_player_lookup_system(df, player_name)

    if player is None:
        return None

    target_birth_date = player["birth_date"]

    print("\nTarget Birth Date:", target_birth_date.date())

    # Find players sharing the exact DOB (including year)
    same_dob = df[
        df["birth_date"] == target_birth_date
    ]

    # Exclude target player
    peers = same_dob[
        same_dob["player"] != player["player"]
    ]

    print("\nPLAYERS WITH EXACT SAME DATE OF BIRTH:")

    if peers.empty:

        print("No other players share the exact date of birth.")

    else:

        print(peers["player"].tolist())

    # Also find players sharing the same calendar birthday
    # regardless of birth year
    same_birthday = df[
        df["birthday"] == target_birth_date.strftime("%m-%d")
    ]

    other_birthday_peers = same_birthday[
        same_birthday["player"] != player["player"]
    ]

    print("\nPLAYERS SHARING THE SAME MONTH AND DAY:")

    if other_birthday_peers.empty:

        print("No other players share this calendar birthday.")

    else:

        print(
            other_birthday_peers[
                ["player", "birth_date"]
            ].to_string(index=False)
        )

    return peers


# ============================================================
# PHASE 5: EVALUATION AND SUMMARY REPORTING
# ============================================================

def phase_5_evaluation_and_summary(df):

    print("\n[PHASE 5] EVALUATION & SUMMARY")
    print("-" * 65)

    total_players = len(df)

    # Unique calendar birthdays
    unique_birthdays = df["birthday"].nunique()

    # Duplicate calendar birthdays
    duplicate_groups = (
        df["birthday"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    # Most common birth month
    month_counts = df["birth_month"].value_counts()

    # Most common birthday
    birthday_counts = df["birthday"].value_counts()

    # Birth year range
    earliest_birth = df["birth_date"].min()
    latest_birth = df["birth_date"].max()

    print("\nFINAL ANALYTICS REPORT")

    print("Total Players:", total_players)

    print("Unique Calendar Birthdays:", unique_birthdays)

    print("Shared Birthday Groups:", len(duplicate_groups))

    print("Players in Duplicate Birthday Groups:",
          df["birthday"].isin(duplicate_groups.index).sum())

    if not month_counts.empty:

        print("Most Common Birth Month:",
              month_counts.idxmax())

        print("Maximum Players Born in One Month:",
              month_counts.max())

    if not birthday_counts.empty:

        print("Most Common Birthday:",
              birthday_counts.idxmax())

        print("Maximum Players Sharing That Birthday:",
              birthday_counts.max())

    print("Earliest Birth Date:", earliest_birth.date())
    print("Latest Birth Date:", latest_birth.date())

    print("\nREPORT COMPLETED!")

    return df


# ============================================================
# FACADE FUNCTION: ORCHESTRATOR
# ============================================================

def run_analytics_lifecycle():

    print("=" * 65)
    print("STARTING NBA BIRTHDAY & BIOGRAPHICAL LIFECYCLE")
    print("=" * 65)

    target_player = "Kareem Abdul-Jabbar"

    # Phase 1
    df = phase_1_discovery_and_sanitization()

    # Phase 2
    df = phase_2_birthday_paradox_analysis(df)

    # Phase 3
    phase_3_player_lookup_system(df, target_player)

    # Phase 4
    phase_4_shared_birthday_cohort(df, target_player)

    # Phase 5
    phase_5_evaluation_and_summary(df)

    print("\n" + "=" * 65)
    print("NBA ANALYTICS LIFECYCLE COMPLETED!")
    print("=" * 65)


# ============================================================
# MAIN GUARD
# ============================================================

if __name__ == "__main__":

    run_analytics_lifecycle()