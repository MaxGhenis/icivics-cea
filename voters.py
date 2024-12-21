import numpy as np
import pandas as pd
from squigglepy.numbers import K, M


def generate_voter_population(params):
    """Generate dataframe of all voters with base characteristics"""
    n_voters = 5 * M  # State population
    n_treated = int(params["marginal_budget"] / params["cost_per_student"])
    years = np.arange(2025, 2045)

    # Generate base characteristics for all voters
    voters = pd.DataFrame(
        {
            "voter_id": np.arange(n_voters),
            "altruism": np.random.beta(2, 5, n_voters),  # Weight on QALYs
            "base_accuracy": np.random.lognormal(
                0.5, 0.5, n_voters
            ),  # QALY perception accuracy
            "base_turnout": np.random.beta(4, 6, n_voters),  # Base turnout probability
            "is_treated": np.arange(n_voters) < n_treated,
            "voter_age": np.random.uniform(
                params["min_age"], params["max_age"], n_voters
            ),
        }
    )

    # Create voter-year pairs
    voter_years = pd.DataFrame(
        [(vid, year) for vid in voters.voter_id for year in years],
        columns=["voter_id", "year"],
    )

    # Merge voter characteristics with years
    voters_full = voter_years.merge(voters, on="voter_id")

    # Calculate years since treatment and years to voting age
    voters_full["years_since_treatment"] = voters_full["year"] - 2025
    voters_full["years_to_vote"] = np.maximum(
        0, 18 - (voters_full["voter_age"] + voters_full["years_since_treatment"])
    )

    return voters_full


def calculate_treatment_effects(voters_df, params):
    """Calculate accuracy and turnout effects for treated population"""
    # Calculate decayed effects
    annual_decay = params["effect_decay"]
    accuracy_boost = params["accuracy_improvement"]
    turnout_boost = params["turnout_boost"]

    decay_factor = annual_decay ** voters_df["years_since_treatment"]

    # Only apply effects to treated voters who are of voting age
    voters_df["is_voting_age"] = voters_df["years_to_vote"] <= 0
    treatment_mask = voters_df["is_treated"] & voters_df["is_voting_age"]

    # Calculate effective accuracy and turnout
    voters_df["accuracy"] = voters_df["base_accuracy"].copy()
    voters_df.loc[treatment_mask, "accuracy"] *= (
        1 + accuracy_boost * decay_factor[treatment_mask]
    )

    voters_df["turnout_prob"] = voters_df["base_turnout"].copy()
    voters_df.loc[treatment_mask, "turnout_prob"] += (
        turnout_boost * decay_factor[treatment_mask]
    )

    return voters_df
