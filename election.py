import numpy as np
import pandas as pd
from squigglepy.numbers import K


def simulate_election_outcomes(voters_df):
    """Simulate election outcomes for all years"""
    # Generate true QALY impacts for candidates
    years = voters_df["year"].unique()
    qaly_impacts = pd.DataFrame(
        {
            "year": years,
            "qalys_a": np.random.lognormal(12, 0.5, len(years)) * K,  # ~160K median
            "qalys_b": np.random.lognormal(10, 0.5, len(years)) * K,  # ~22K median
        }
    )

    # Simulate turnout
    voters_df["votes"] = np.random.random(len(voters_df)) < voters_df["turnout_prob"]

    # For voters who turn out, simulate their perception of QALYs and voting choice
    voting_mask = voters_df["votes"]
    if voting_mask.sum() > 0:
        # Generate noise in QALY perception (inverse relationship with accuracy)
        noise_scale = 1 / voters_df.loc[voting_mask, "accuracy"]
        noise_a = np.random.normal(0, noise_scale)
        noise_b = np.random.normal(0, noise_scale)

        # Generate other (non-QALY) preferences
        other_prefs_a = np.random.normal(0, 1, voting_mask.sum())
        other_prefs_b = np.random.normal(0, 1, voting_mask.sum())

        # Merge QALY impacts with voters
        voters_temp = voters_df[voting_mask].merge(qaly_impacts, on="year")

        # Calculate utilities including noise and other preferences
        perceived_qalys_a = voters_temp["qalys_a"] * (1 + noise_a)
        perceived_qalys_b = voters_temp["qalys_b"] * (1 + noise_b)

        utility_a = (
            voters_temp["altruism"] * perceived_qalys_a
            + (1 - voters_temp["altruism"]) * other_prefs_a
        )
        utility_b = (
            voters_temp["altruism"] * perceived_qalys_b
            + (1 - voters_temp["altruism"]) * other_prefs_b
        )

        voters_df.loc[voting_mask, "vote_a"] = utility_a > utility_b
    else:
        voters_df["vote_a"] = False

    # Aggregate results by year
    results = pd.DataFrame()
    results["total_votes"] = voters_df.groupby("year")["votes"].sum()
    results["votes_a"] = voters_df[voters_df["votes"]].groupby("year")["vote_a"].sum()
    results["total_valid_votes"] = (
        voters_df[voters_df["votes"]].groupby("year")["vote_a"].count()
    )

    # Reset index to make year a column
    results = results.reset_index()

    # Merge with QALY impacts
    results = results.merge(qaly_impacts, on="year")

    # Calculate winner and QALY impact
    results["vote_share_a"] = results["votes_a"] / results["total_valid_votes"]
    results["winner_qalys"] = np.where(
        results["vote_share_a"] > 0.5, results["qalys_a"], results["qalys_b"]
    )

    return results
