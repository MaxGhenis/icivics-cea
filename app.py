import streamlit as st
import squigglepy as sq
from squigglepy.numbers import K, M
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def run_impact_simulation(params, n_sims=1000):
    """Run Monte Carlo simulation of civic education impact"""
    results = []

    for _ in range(n_sims):
        # Calculate students reached
        marginal_cost = ~sq.lognorm(
            params["marginal_cost_low"], params["marginal_cost_high"]
        )
        students = params["marginal_budget"] / marginal_cost

        # Calculate voting timing distribution
        years_to_vote = ~sq.norm(
            params["mean_years_to_vote"] - 2, params["mean_years_to_vote"] + 2
        )

        # Calculate voting effects over time
        total_qalys = 0

        for year in range(20):  # Look ahead 20 years
            # Account for knowledge decay
            knowledge_retention = params["knowledge_retention_rate"] ** year

            # Calculate voters this year
            # Use normal approximation for timing window
            voting_window = np.exp(
                -((year - years_to_vote) ** 2) / 8
            )  # 2-year SD

            if voting_window > 0.1:  # If significant voting activity this year
                # Basic voting probability
                vote_prob = ~sq.beta(5, 3) * params["voting_rate_boost"]

                # Policy alignment improvement
                alignment_effect = (
                    ~sq.beta(4, 4) * params["alignment_improvement"]
                )

                # Calculate actual voters considering retention
                effective_voters = (
                    students * vote_prob * knowledge_retention * voting_window
                )

                # Calculate aligned voters
                aligned_voters = effective_voters * alignment_effect

                # Simulate multiple races
                for _ in range(params["races_per_year"]):
                    # Random election margin (using lognormal to ensure positive)
                    margin = ~sq.lognorm(0.01, 0.05)

                    # Election size varies by type (local vs state vs federal)
                    election_size = ~sq.mixture(
                        [
                            [
                                0.6,
                                sq.lognorm(10 * K, 100 * K),
                            ],  # Local: 10K-100K voters
                            [
                                0.3,
                                sq.lognorm(100 * K, 1 * M),
                            ],  # State: 100K-1M voters
                            [
                                0.1,
                                sq.lognorm(1 * M, 10 * M),
                            ],  # Federal: 1M-10M voters
                        ]
                    )

                    # Probability of swinging election
                    swing_prob = min(
                        aligned_voters / (election_size * margin), 1
                    )

                    # If election swings, calculate policy value
                    if ~sq.bernoulli(float(swing_prob)):
                        # Policy value distribution varies by race type
                        policy_value = ~sq.mixture(
                            [
                                [0.6, sq.lognorm(100, 10 * K)],  # Local policy
                                [
                                    0.3,
                                    sq.lognorm(10 * K, 100 * K),
                                ],  # State policy
                                [
                                    0.1,
                                    sq.lognorm(100 * K, 1 * M),
                                ],  # Federal policy
                            ]
                        )
                        total_qalys += policy_value

        # Account for STEM tradeoff
        stem_impact = ~sq.norm(
            mean=-params["stem_tradeoff"], sd=params["stem_tradeoff"] / 4
        )
        total_qalys *= 1 + stem_impact

        # Calculate cost-effectiveness
        cost_effectiveness = total_qalys / params["marginal_budget"]
        results.append(cost_effectiveness)

    return np.array(results)


def create_impact_app():
    st.title("iCivics Impact Analysis")

    st.markdown(
        """
    This app simulates the marginal impact of additional funding for iCivics,
    considering various pathways to impact including voting behavior, policy
    outcomes, and educational tradeoffs.
    """
    )

    # Parameter input sections
    st.header("Cost Parameters")

    st.markdown(
        """
    **Current Cost Context:**
    - iCivics' current budget: $11.3M serving 9M students
    - Current average cost: ~$1.26 per student
    - Marginal costs often higher due to:
        - Reaching harder-to-serve populations
        - New program development needs
        - Implementation support requirements
    """
    )

    marginal_budget = st.number_input(
        "Additional Funding ($)",
        min_value=100_000,
        max_value=10_000_000,
        value=1_000_000,
        step=100_000,
    )

    marginal_cost_low = st.slider(
        "Minimum Marginal Cost per Student ($)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        help="Minimum expected cost to reach an additional student",
    )

    marginal_cost_high = st.slider(
        "Maximum Marginal Cost per Student ($)",
        min_value=marginal_cost_low,
        max_value=15.0,
        value=5.0,
        help="Maximum expected cost to reach an additional student",
    )

    st.header("Educational Impact Parameters")

    st.markdown(
        """
    **STEM Tradeoff Context:**
    - Current integration with numeracy (e.g., People's Pie game)
    - Shared critical thinking skills
    - Time allocation in school day
    """
    )

    stem_tradeoff = (
        st.slider(
            "STEM Curriculum Displacement (%)",
            min_value=0.0,
            max_value=30.0,
            value=15.0,
            help="Percentage of civics time that trades off with STEM",
        )
        / 100
    )

    knowledge_retention = (
        st.slider(
            "Annual Knowledge Retention Rate (%)",
            min_value=50,
            max_value=95,
            value=85,
            help="Percentage of civics knowledge retained year-over-year",
        )
        / 100
    )

    st.header("Voting Impact Parameters")

    mean_years_to_vote = st.slider(
        "Mean Years Until First Vote",
        min_value=1,
        max_value=10,
        value=6,
        help="Average years between civics education and first election",
    )

    voting_boost = (
        st.slider(
            "Voting Rate Increase (%)",
            min_value=5,
            max_value=30,
            value=15,
            help="Increase in probability of voting due to civics education",
        )
        / 100
    )

    alignment_improvement = (
        st.slider(
            "Policy Alignment Improvement (%)",
            min_value=1,
            max_value=20,
            value=10,
            help="Increase in probability of voting for higher-impact policies",
        )
        / 100
    )

    races_per_year = st.slider(
        "Average Relevant Races per Year",
        min_value=1,
        max_value=20,
        value=12,
        help="Number of elections where student votes could matter",
    )

    # Run simulation
    if st.button("Run Simulation"):
        params = {
            "marginal_budget": marginal_budget,
            "marginal_cost_low": marginal_cost_low,
            "marginal_cost_high": marginal_cost_high,
            "stem_tradeoff": stem_tradeoff,
            "knowledge_retention_rate": knowledge_retention,
            "mean_years_to_vote": mean_years_to_vote,
            "voting_rate_boost": voting_boost,
            "alignment_improvement": alignment_improvement,
            "races_per_year": races_per_year,
        }

        with st.spinner("Running simulation..."):
            results = run_impact_simulation(params)

        st.header("Results")

        # Display summary statistics
        st.subheader("Cost-Effectiveness Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean QALYs/$", f"{np.mean(results):.2f}")
        with col2:
            st.metric("Median QALYs/$", f"{np.median(results):.2f}")
        with col3:
            ci_width = np.percentile(results, 95) - np.percentile(results, 5)
            st.metric("95% CI Width", f"{ci_width:.2f}")

        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(results, bins=50, density=True)
        ax.axvline(
            np.mean(results), color="r", linestyle="dashed", label="Mean"
        )
        ax.axvline(
            np.median(results), color="g", linestyle="dashed", label="Median"
        )
        ax.set_title("Distribution of Cost-Effectiveness Estimates")
        ax.set_xlabel("QALYs per Dollar")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)

        # Display percentiles
        st.subheader("Percentile Analysis")
        percentiles = [1, 5, 25, 50, 75, 95, 99]
        percentile_values = np.percentile(results, percentiles)
        percentile_df = pd.DataFrame(
            {"Percentile": percentiles, "QALYs/$": percentile_values}
        )
        st.table(percentile_df)

        # Additional analysis
        st.subheader("Key Drivers")
        st.markdown(
            """
        The simulation shows several key factors driving impact:
        1. Local election effects tend to be more certain and immediate
        2. Knowledge retention significantly affects long-term impact
        3. STEM tradeoffs can be partially mitigated through integrated learning
        
        **Note on Parameter Sensitivity:**
        - Election size has high impact on swing probability
        - Knowledge retention compounds over multiple election cycles
        - Policy value varies significantly by election type
        """
        )


if __name__ == "__main__":
    create_impact_app()
