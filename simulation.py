import streamlit as st
import squigglepy as sq
from squigglepy.numbers import K, M
import numpy as np
import pandas as pd
from voters import generate_voter_population, calculate_treatment_effects
from election import simulate_election_outcomes


def simulate_civics_impact(params):
    """Run full simulation comparing baseline to treatment"""
    with st.spinner("Generating voter population..."):
        voters = generate_voter_population(params)

    with st.spinner("Calculating treatment effects..."):
        voters_with_effects = calculate_treatment_effects(voters, params)

    with st.spinner("Simulating elections..."):
        # Run baseline simulation (setting treatment effects to 0)
        baseline_voters = voters.copy()
        baseline_voters["accuracy"] = baseline_voters["base_accuracy"]
        baseline_voters["turnout_prob"] = baseline_voters["base_turnout"]
        baseline_results = simulate_election_outcomes(baseline_voters)

        # Run treatment simulation
        treatment_results = simulate_election_outcomes(voters_with_effects)

    # Calculate differences
    results = pd.DataFrame(
        {
            "year": baseline_results["year"],
            "qaly_diff": treatment_results["winner_qalys"]
            - baseline_results["winner_qalys"],
            "turnout_diff": (
                treatment_results["total_votes"]
                / len(voters_with_effects.groupby("year"))
                - baseline_results["total_votes"] / len(baseline_voters.groupby("year"))
            ),
        }
    )

    return results
