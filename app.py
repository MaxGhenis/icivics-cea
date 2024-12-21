######################################
# Streamlit + Squigglepy Civics QALY Demo
######################################
import streamlit as st
import squigglepy as sq
from squigglepy import bayes, numbers
import numpy as np
import statistics
import matplotlib.pyplot as plt

# We do need to enable "streamlit run" to visualize the app:
#   streamlit run this_file.py


def simulate_icivics_qaly(
    n_samples,
    cost_per_child_dist,
    # child_count_dist,  # optional if we want to randomize total children
    extra_funds,
    # Probability that civics education shifts a child’s vote to a "more beneficial" outcome
    p_vote_shift_dist,
    # Probability it shifts in the "wrong" direction
    p_vote_negative_dist,
    # Range for how many net votes are needed to tip an election
    margin_of_victory_dist,
    # QALY difference if the "better" candidate or policy wins
    qaly_gain_dist,
    # Distribution for how many children we think can vote in T years (discounting)
    years_to_vote_dist,
    discount_rate,
    # Substitution effect: crowding out more valuable instruction
    p_substitute_dist,
    substitution_impact_dist,
    # Complement effect: iCivics numeracy or synergy
    p_complement_dist,
    complement_boost_dist,
    n_elections_dist,
    continuity_prob_dist
):
    """
    Monte Carlo simulation using Squigglepy distributions.

    Returns an array or list of 'QALY gains' for each draw.
    """
    # 1) Sample each parameter n_samples times from the distributions.
    #    Squigglepy uses the '@' operator to get multiple samples.
    cost_per_child = cost_per_child_dist @ n_samples
    p_vote_shift = p_vote_shift_dist @ n_samples
    p_vote_negative = p_vote_negative_dist @ n_samples
    margin_of_victory = margin_of_victory_dist @ n_samples
    qaly_gain = qaly_gain_dist @ n_samples
    years_to_vote = years_to_vote_dist @ n_samples
    p_substitute = p_substitute_dist @ n_samples
    substitution_impact = substitution_impact_dist @ n_samples
    p_complement = p_complement_dist @ n_samples
    complement_boost = complement_boost_dist @ n_samples
    n_elections = n_elections_dist @ n_samples
    continuity_prob = continuity_prob_dist @ n_samples

    results = []

    for i in range(n_samples):
        # 2) Determine how many children we can reach with the extra funds
        cpc = cost_per_child[i]
        # Potentially we might clamp cost to a floor if it is near 0, but here we assume it's in 1.5-3.0 range
        if cpc <= 0:
            # Just skip or set cpc to 0.01 to avoid division by zero
            cpc = 0.01

        children_educated = extra_funds / cpc

        # 3) Probability that a child's vote shifts "positively" vs "negatively"
        p_pos = p_vote_shift[i]
        p_neg = p_vote_negative[i]
        net_positive_voters = children_educated * p_pos
        net_negative_voters = children_educated * p_neg
        net_votes_for_superior = net_positive_voters - net_negative_voters

        # 4) Check margin_of_victory to see if we flip an election
        mov = margin_of_victory[i]
        election_flipped = (net_votes_for_superior >= (mov / 2.0))

        # 5) If flipped, we get QALY difference
        qgain = 0.0
        if election_flipped:
            # Multi-election logic
            #   - first election at T = years_to_vote[i] (discount factor)
            #   - subsequent ones happen at 4-year intervals
            #   - each subsequent election outcome persists with continuity_prob
            #   - e.g. if continuity_prob=0.8, there's an 80% chance the same outcome holds next time

            # discount factor for first election
            Y = years_to_vote[i]
            if Y < 0:
                Y = 0  # clamp in case we got negative draws

            base_q = qaly_gain[i]  # the difference in QALYs vs the inferior outcome
            q0 = base_q / ((1 + discount_rate) ** Y)
            qgain += q0

            # subsequent elections:
            election_interval = 4
            nelec = int(n_elections[i])
            cp = continuity_prob[i]
            for e in range(1, nelec):
                # Probability that beneficial outcome persists at election e
                # We'll do a simple approach: each cycle is "kept" with probability = cp^(e)
                # or we do repeated draws. We'll do the exponent approach:
                if np.random.rand() < (cp ** e):
                    # discount by Y + e*(4 years)
                    years_elapsed = Y + e * election_interval
                    qgain += base_q / ((1 + discount_rate) ** years_elapsed)

        # 6) Substitution vs. complement
        sub_frac = p_substitute[i]
        sub_impact = substitution_impact[i]
        penalty_factor = 1.0 - (sub_frac * sub_impact)

        comp_frac = p_complement[i]
        comp_boost = complement_boost[i]
        boost_factor = 1.0 + (comp_frac * comp_boost)

        final_q = qgain * penalty_factor * boost_factor
        results.append(final_q)

    return np.array(results)

def main():
    st.title("iCivics Civics Education QALY Impact Simulation (with Squigglepy)")
    st.markdown(
        """
        This app models how **marginal spending** on iCivics (or a similar civics 
        education program) might affect future electoral outcomes **in QALY terms**. 
        All parameters are uncertain and can be adjusted below.
        """
    )

    n_samples = st.sidebar.slider(
        "Number of Monte Carlo Samples",
        min_value=1_000,
        max_value=300_000,
        value=50_000,
        step=1_000,
    )

    extra_funds = st.sidebar.number_input(
        "Extra Funds (USD) for Marginal Civics Education",
        value=1_000_000,
        step=100_000,
    )

    discount_rate = st.sidebar.slider(
        "Discount Rate (annual)",
        min_value=0.0,
        max_value=0.10,
        value=0.04,
        step=0.01,
    )

    st.subheader("Parameter Ranges & Justifications")

    st.markdown(
        """
        **Marginal Cost per Additional Child**  
        The average cost per student has historically hovered around \$2, 
        but at the margin (i.e., with more funding), we might see it 
        **increase** (due to diminishing returns) or **decrease** 
        (due to economies of scale). A plausible range is \$1.5–\$3.
        """
    )
    cost_low = st.slider("Cost Per Child - Low", 0.50, 5.0, 1.5, 0.1)
    cost_high = st.slider("Cost Per Child - High", cost_low, 6.0, 3.0, 0.1)
    cost_per_child_dist = sq.uniform(cost_low, cost_high)

    st.markdown(
        """
        **Probability of Positive Vote Shift**  
        The fraction of newly educated students whose vote swings 
        *toward* the higher-value candidate or policy.  
        """
    )
    p_vote_shift_low = st.slider("p_vote_shift (low)", 0.0, 0.5, 0.01, 0.01)
    p_vote_shift_high = st.slider("p_vote_shift (high)", p_vote_shift_low, 0.7, 0.10, 0.01)
    p_vote_shift_dist = sq.uniform(p_vote_shift_low, p_vote_shift_high)

    st.markdown(
        """
        **Probability of Negative Vote Shift**  
        A small fraction might interpret civics in a way that 
        reduces net benefits (counterfactual behavior).  
        """
    )
    p_vote_neg_low = st.slider("p_vote_negative (low)", 0.0, 0.1, 0.0, 0.01)
    p_vote_neg_high = st.slider("p_vote_negative (high)", p_vote_neg_low, 0.1, 0.02, 0.01)
    p_vote_negative_dist = sq.uniform(p_vote_neg_low, p_vote_neg_high)

    st.markdown(
        """
        **Margin of Victory**  
        Typical elections can be decided by anywhere from a few 
        thousand votes to tens of thousands.  
        """
    )
    mov_low = st.slider("Margin of Victory - Low", 1_000.0, 100_000.0, 5_000.0, 1_000.0)
    mov_high = st.slider("Margin of Victory - High", mov_low, 200_000.0, 50_000.0, 1_000.0)
    margin_of_victory_dist = sq.uniform(mov_low, mov_high)

    st.markdown(
        """
        **QALY Difference**  
        How big is the difference in societal impact (in QALYs) 
        between the "superior" and "inferior" outcome?  
        This can range widely—from small local shifts (thousands of QALYs) 
        to large-scale policy changes (hundreds of thousands or millions).
        """
    )
    qaly_low = st.slider("QALY difference - Low", 0.0, 1e7, 1e6, 1e5)
    qaly_high = st.slider("QALY difference - High", qaly_low, 1e9, 1e8, 1e7)
    qaly_gain_dist = sq.uniform(qaly_low, qaly_high)

    st.markdown(
        """
        **Years Until Children Vote**  
        The time-lag from civics education to first voting: 
        could be 1 year (if in high school) or up to 8-10 years 
        (if younger).  
        """
    )
    years_low = st.slider("Years to Vote - Low", 0, 10, 1, 1)
    years_high = st.slider("Years to Vote - High", years_low, 15, 10, 1)
    years_to_vote_dist = sq.uniform(years_low, years_high)

    st.markdown(
        """
        **Substitution vs Complement**  
        - Substitution: Some fraction of students might lose 
          more valuable STEM or reading time, partially offsetting 
          the benefit.  
        - Complement: Some fraction might get extra numeracy 
          from iCivics' budgeting or data-driven components, 
          boosting the net effect.
        """
    )
    sub_frac_low = st.slider("p_substitute (low)", 0.0, 0.5, 0.0, 0.01)
    sub_frac_high = st.slider("p_substitute (high)", sub_frac_low, 1.0, 0.3, 0.01)
    p_substitute_dist = sq.uniform(sub_frac_low, sub_frac_high)

    sub_impact_low = st.slider("substitution_impact (low)", 0.0, 1.0, 0.0, 0.05)
    sub_impact_high = st.slider("substitution_impact (high)", sub_impact_low, 1.0, 0.5, 0.05)
    substitution_impact_dist = sq.uniform(sub_impact_low, sub_impact_high)

    comp_frac_low = st.slider("p_complement (low)", 0.0, 1.0, 0.0, 0.05)
    comp_frac_high = st.slider("p_complement (high)", comp_frac_low, 1.0, 0.3, 0.05)
    p_complement_dist = sq.uniform(comp_frac_low, comp_frac_high)

    comp_boost_low = st.slider("complement_boost (low)", 0.0, 1.0, 0.0, 0.05)
    comp_boost_high = st.slider("complement_boost (high)", comp_boost_low, 1.0, 0.3, 0.05)
    complement_boost_dist = sq.uniform(comp_boost_low, comp_boost_high)

    st.markdown(
        """
        **Number of Elections & Continuity**  
        - Because policy or incumbency might persist, 
          flipping one election may yield multiple election cycles 
          of improved outcomes.  
        """
    )
    nelec_low = st.slider("n_elections (low)", 1, 5, 1, 1)
    nelec_high = st.slider("n_elections (high)", nelec_low, 10, 3, 1)
    n_elections_dist = sq.uniform(nelec_low, nelec_high)

    cont_prob_low = st.slider("continuity_prob (low)", 0.0, 1.0, 0.5, 0.05)
    cont_prob_high = st.slider("continuity_prob (high)", cont_prob_low, 1.0, 1.0, 0.05)
    continuity_prob_dist = sq.uniform(cont_prob_low, cont_prob_high)

    # ----- Run Simulation -----
    st.write("## Simulation in Progress...")

    qaly_samples = simulate_icivics_qaly(
        n_samples,
        cost_per_child_dist,
        extra_funds,
        p_vote_shift_dist,
        p_vote_negative_dist,
        margin_of_victory_dist,
        qaly_gain_dist,
        years_to_vote_dist,
        discount_rate,
        p_substitute_dist,
        substitution_impact_dist,
        p_complement_dist,
        complement_boost_dist,
        n_elections_dist,
        continuity_prob_dist,
    )

    mean_val = np.mean(qaly_samples)
    median_val = np.median(qaly_samples)
    p5 = np.percentile(qaly_samples, 5)
    p95 = np.percentile(qaly_samples, 95)

    st.write(
        f"""
        **Results** (based on {n_samples} draws):
        - **Mean QALYs**: {mean_val:,.2f}
        - **Median QALYs**: {median_val:,.2f}
        - **5th percentile**: {p5:,.2f}
        - **95th percentile**: {p95:,.2f}
        - **Mean QALYs per $**: {mean_val / extra_funds:,.2f}
        """
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(qaly_samples, bins=100, color="skyblue", edgecolor="black")
    ax.set_title("Distribution of QALY Gains")
    ax.set_xlabel("QALY Gains")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


if __name__ == "__main__":
    main()