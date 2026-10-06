from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AML Monitoring", layout="wide")

@st.cache_data
def load_alerts():
    path = Path(__file__).parent / "validation_alerts.csv"

    return pd.read_csv(
        path,
        dtype={"Account": str, "Account.1": str}
    ).sort_values("risk_score", ascending=False).reset_index(drop=True)

alerts = load_alerts()

st.title("AML Transaction Monitoring")
st.caption(
    "Random forest • Synthetic validation data • "
    "Retrospective review simulation"
)

st.sidebar.header("Review capacity")

budget = int(st.sidebar.number_input(
    "Transactions to review",
    min_value=1,
    max_value=len(alerts),
    value=min(100, len(alerts)),
    step=1
))

selected = alerts.head(budget)

caught = int(selected["Is Laundering"].sum())
total_positive = int(alerts["Is Laundering"].sum())
precision = caught / budget
recall = caught / total_positive if total_positive else 0

a, b, c, d = st.columns(4)

a.metric("Transactions reviewed", f"{budget:,}")
b.metric("Labelled laundering caught", f"{caught:,}")
c.metric("Precision at budget", f"{precision:.1%}")
d.metric("Recall at budget", f"{recall:.1%}")

st.caption(
    "The budget applies to the full validation period, not one day. "
    "Scores are rankings, not calibrated laundering probabilities."
)

st.subheader("Ranked alert queue")

columns = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Paid",
    "Payment Currency",
    "risk_score",
    "review_context"
]

show_labels = st.checkbox("Show known labels for evaluation")

if show_labels:
    columns.append("Is Laundering")

st.dataframe(
    selected[columns],
    use_container_width=True,
    hide_index=True
)

st.subheader("Inspect a transaction")

rank = int(st.number_input(
    "Alert rank",
    min_value=1,
    max_value=budget,
    value=1
))

transaction = selected.iloc[rank - 1]

st.write(
    f"**Sender:** {transaction['From Bank']} / "
    f"{transaction['Account']}"
)

st.write(
    f"**Receiver:** {transaction['To Bank']} / "
    f"{transaction['Account.1']}"
)

st.write(
    f"**Amount:** {transaction['Amount Paid']:,.2f} "
    f"{transaction['Payment Currency']}"
)

st.write(f"**Model score:** {transaction['risk_score']:.4f}")
st.info(transaction["review_context"])

st.caption(
    "Review context describes transaction attributes. "
    "It does not explain the model's exact reasoning or establish wrongdoing."
)
