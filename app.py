import streamlit as st
from simulation import simulate_civics_impact
import matplotlib.pyplot as plt
import numpy as np


def create_impact_app():
    st.title("iCivics Impact Analysis - Vectorized Model")

    st.markdown(
        """
    This simulation models individual voter behavior based on:
    - Altruism (weight assigned to QALYs vs other preferences)
    - Accuracy (ability to perceive true QALY impacts)
    - Turnout probability
    """
    )

    # Parameters
    st.header("Parameters")

    col1, col2 = st.columns(2)

    with col1:
        marginal_budget = st.number_input(
            "Additional Funding ($)",
            min_value=100_000,
            max_value=10_000_000,
            value=1_000_000,
        )

        cost_per_student = st.number_input(
            "Cost per Student ($)", min_value=1.0, max_value=20.0, value=5.0
        )

        accuracy_improvement = st.slider(
            "Accuracy Improvement",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            help="Proportional improvement in QALY perception accuracy",
        )

    with col2:
        turnout_boost = st.slider(
            "Turnout Boost",
            min_value=0.0,
            max_value=0.3,
            value=0.1,
            help="Percentage point increase in turnout probability",
        )

        effect_decay = st.slider(
            "Annual Effect Decay",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            help="Proportion of effect retained each year",
        )

        min_age = st.number_input("Minimum Student Age", value=12)
        max_age = st.number_input("Maximum Student Age", value=18)

    if st.button("Run Simulation"):
        params = {
            "marginal_budget": marginal_budget,
            "cost_per_student": cost_per_student,
            "accuracy_improvement": accuracy_improvement,
            "turnout_boost": turnout_boost,
            "effect_decay": effect_decay,
            "min_age": min_age,
            "max_age": max_age,
        }

        results = simulate_civics_impact(params)

        # Calculate cost-effectiveness
        total_qaly_diff = results["qaly_diff"].sum()
        cost_effectiveness = total_qaly_diff / marginal_budget

        st.header("Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total QALYs", f"{total_qaly_diff:,.0f}")
        with col2:
            st.metric("QALYs per Dollar", f"{cost_effectiveness:.2f}")

        # Plot QALY difference over time
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results["year"], results["qaly_diff"])
        ax.set_title("QALY Impact by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("QALY Difference")
        st.pyplot(fig)

        # Plot turnout difference
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results["year"], results["turnout_diff"])
        ax.set_title("Turnout Difference by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage Point Difference")
        st.pyplot(fig)

        # Show detailed results
        st.subheader("Detailed Results by Year")
        st.dataframe(results)


if __name__ == "__main__":
    create_impact_app()
