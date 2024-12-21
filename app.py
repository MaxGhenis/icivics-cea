import streamlit as st
import squigglepy as sq
from squigglepy.numbers import K, M
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Voter:
    def __init__(self, altruism, accuracy, baseline_turnout_prob):
        self.altruism = altruism  # Weight on QALYs vs other preferences (0-1)
        self.accuracy = (
            accuracy  # How well they perceive true QALYs (higher = less noise)
        )
        self.baseline_turnout_prob = baseline_turnout_prob

    def perceive_qalys(self, true_qalys):
        """Add noise to QALY perception based on accuracy"""
        noise = np.random.normal(0, 1 / self.accuracy)
        return true_qalys + noise

    def vote_choice(self, candidate_qalys, other_preferences):
        """Return preferred candidate based on QALYs and other preferences"""
        perceived_qalys = {
            k: self.perceive_qalys(v) for k, v in candidate_qalys.items()
        }

        utilities = {}
        for candidate in candidate_qalys.keys():
            utilities[candidate] = (
                self.altruism * perceived_qalys[candidate]
                + (1 - self.altruism) * other_preferences[candidate]
            )

        return max(utilities.items(), key=lambda x: x[1])[0]

    def will_vote(self, civics_boost=0):
        """Determine if voter turns out"""
        return np.random.random() < (self.baseline_turnout_prob + civics_boost)


class Election:
    def __init__(self, year, state_pop=5 * M):
        self.year = year
        self.state_pop = state_pop
        self.eligible_voters = []

    def add_voter(self, voter):
        self.eligible_voters.append(voter)

    def get_qaly_impacts(self, term_years=4, decay_rate=0.8):
        """Generate QALY impacts for candidates over time"""
        # Candidate A: Higher QALY impact
        qalys_a = {}
        base_impact_a = ~sq.lognorm(50 * K, 200 * K)  # 50K-200K QALYs in first year
        for y in range(self.year, self.year + 20):
            if y < self.year + term_years:
                qalys_a[y] = base_impact_a * (decay_rate ** (y - self.year))
            else:
                # Post-term effects decay faster
                qalys_a[y] = base_impact_a * (decay_rate ** (2 * (y - self.year)))

        # Candidate B: Lower QALY impact
        qalys_b = {}
        base_impact_b = ~sq.lognorm(10 * K, 50 * K)  # 10K-50K QALYs in first year
        for y in range(self.year, self.year + 20):
            if y < self.year + term_years:
                qalys_b[y] = base_impact_b * (decay_rate ** (y - self.year))
            else:
                qalys_b[y] = base_impact_b * (decay_rate ** (2 * (y - self.year)))

        return {"A": qalys_a, "B": qalys_b}

    def run_election(self, civics_boost=0):
        """Run election and return results"""
        votes = {"A": 0, "B": 0}
        turnout = 0

        # Get QALY impacts for candidates
        candidate_qalys = self.get_qaly_impacts()
        total_qalys_by_candidate = {
            k: sum(v.values()) for k, v in candidate_qalys.items()
        }

        for voter in self.eligible_voters:
            if voter.will_vote(civics_boost):
                turnout += 1
                # Generate random other preferences
                other_prefs = {"A": np.random.normal(0, 1), "B": np.random.normal(0, 1)}
                choice = voter.vote_choice(total_qalys_by_candidate, other_prefs)
                votes[choice] += 1

        winner = max(votes.items(), key=lambda x: x[1])[0]
        margin = abs(votes["A"] - votes["B"]) / sum(votes.values())

        return {
            "winner": winner,
            "margin": margin,
            "turnout": turnout / len(self.eligible_voters),
            "votes": votes,
            "qaly_impact": candidate_qalys[winner],
        }


def simulate_civics_impact(params):
    """Simulate impact of marginal civics funding over time"""
    results = []

    # Generate baseline population parameters
    baseline_altruism = ~sq.beta(2, 5)  # Right-skewed, mean around 0.3
    baseline_accuracy = ~sq.lognorm(0.5, 2)  # How well voters perceive QALYs
    baseline_turnout = ~sq.beta(4, 6)  # Right-skewed, mean around 0.4

    # Effect sizes from civics education
    accuracy_boost = params["accuracy_improvement"]
    turnout_boost = params["turnout_boost"]

    # Effect decay
    annual_decay = params["effect_decay"]

    # Run simulations for each year
    progress_bar = st.progress(0)
    for i, year in enumerate(range(2025, 2045)):
        # Create election
        election_base = Election(year)
        election_treatment = Election(year)

        # Add voters
        n_voters = 5 * M  # State population
        n_treated = int(params["marginal_budget"] / params["cost_per_student"])

        # Create a second progress bar for voter simulation
        voter_progress = st.progress(0)

        # Process voters in batches for efficiency
        batch_size = 10000
        for batch_i in range(0, n_voters, batch_size):
            # Update progress bar every batch
            voter_progress.progress(min(1.0, (batch_i + batch_size) / n_voters))

            batch_end = min(batch_i + batch_size, n_voters)
            for i in range(batch_i, batch_end):
                # Basic voter characteristics
                altruism = ~sq.beta(2, 5)
                accuracy = ~sq.lognorm(0.5, 2)
                turnout_prob = ~sq.beta(4, 6)

                # Add to baseline election
                election_base.add_voter(Voter(altruism, accuracy, turnout_prob))

                # Add to treatment election with boosted parameters for treated population
                if i < n_treated:
                    # Apply civics education effects with decay
                    years_since_treatment = max(0, year - 2025)
                    current_accuracy_boost = accuracy_boost * (
                        annual_decay**years_since_treatment
                    )
                    current_turnout_boost = turnout_boost * (
                        annual_decay**years_since_treatment
                    )

                    # Check if voter is of voting age
                    voter_age = np.random.uniform(params["min_age"], params["max_age"])
                    years_to_vote = max(0, 18 - voter_age)

                    if years_since_treatment >= years_to_vote:
                        election_treatment.add_voter(
                            Voter(
                                altruism,
                                accuracy * (1 + current_accuracy_boost),
                                turnout_prob + current_turnout_boost,
                            )
                        )
                    else:
                        election_treatment.add_voter(
                            Voter(altruism, accuracy, turnout_prob)
                        )
                else:
                    election_treatment.add_voter(
                        Voter(altruism, accuracy, turnout_prob)
                    )
            # Basic voter characteristics
            altruism = ~sq.beta(2, 5)
            accuracy = ~sq.lognorm(0.5, 2)
            turnout_prob = ~sq.beta(4, 6)

            # Add to baseline election
            election_base.add_voter(Voter(altruism, accuracy, turnout_prob))

            # Add to treatment election with boosted parameters for treated population
            if i < n_treated:
                # Apply civics education effects with decay
                years_since_treatment = max(0, year - 2025)
                current_accuracy_boost = accuracy_boost * (
                    annual_decay**years_since_treatment
                )
                current_turnout_boost = turnout_boost * (
                    annual_decay**years_since_treatment
                )

                # Check if voter is of voting age
                voter_age = np.random.uniform(params["min_age"], params["max_age"])
                years_to_vote = max(0, 18 - voter_age)

                if years_since_treatment >= years_to_vote:
                    election_treatment.add_voter(
                        Voter(
                            altruism,
                            accuracy * (1 + current_accuracy_boost),
                            turnout_prob + current_turnout_boost,
                        )
                    )
                else:
                    election_treatment.add_voter(
                        Voter(altruism, accuracy, turnout_prob)
                    )
            else:
                election_treatment.add_voter(Voter(altruism, accuracy, turnout_prob))

        # Run elections
        base_result = election_base.run_election()
        treatment_result = election_treatment.run_election()

        # Calculate QALY difference
        qaly_diff = sum(treatment_result["qaly_impact"].values()) - sum(
            base_result["qaly_impact"].values()
        )

        # Update progress bar
        progress_bar.progress((i + 1) / 20)

        results.append(
            {
                "year": year,
                "qaly_diff": qaly_diff,
                "base_turnout": base_result["turnout"],
                "treatment_turnout": treatment_result["turnout"],
                "base_margin": base_result["margin"],
                "treatment_margin": treatment_result["margin"],
            }
        )

    return pd.DataFrame(results)


def create_impact_app():
    st.title("iCivics Impact Analysis - Micro-founded Model")

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
        ax.plot(results["year"], results["treatment_turnout"] - results["base_turnout"])
        ax.set_title("Turnout Difference by Year")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage Point Difference")
        st.pyplot(fig)

        # Show detailed results
        st.subheader("Detailed Results by Year")
        st.dataframe(results)


if __name__ == "__main__":
    create_impact_app()
