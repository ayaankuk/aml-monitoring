"""Explore notebook exports: python -m streamlit run app/dashboard.py."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AML Transaction Monitoring", page_icon="🔎", layout="wide")

DATA_DIR = Path(__file__).parent


@st.cache_data
def load_alerts(path, modified_ns):
    """File modification time invalidates the cache after a notebook rerun."""
    alerts = pd.read_csv(
        path,
        dtype={"Account": str, "Account.1": str, "From Bank": str, "To Bank": str},
        parse_dates=["Timestamp"],
        float_precision="round_trip",
    )
    required = {
        "transaction_id", "Timestamp", "From Bank", "Account", "To Bank",
        "Account.1", "Amount Paid", "Payment Currency", "Payment Format",
        "risk_score", "review_context", "Is Laundering",
    }
    missing = required.difference(alerts.columns)
    if missing:
        raise ValueError(f"Export is missing columns: {', '.join(sorted(missing))}")
    if alerts.empty:
        raise ValueError("The selected export contains no transactions.")
    if alerts["transaction_id"].isna().any() or alerts["transaction_id"].duplicated().any():
        raise ValueError("Transaction IDs must be present and unique.")
    if not alerts["risk_score"].between(0, 1).all():
        raise ValueError("Risk scores must be finite numbers between 0 and 1.")
    if not alerts["Is Laundering"].isin([0, 1]).all():
        raise ValueError("Evaluation labels must be 0 or 1.")
    return alerts.sort_values(
        ["risk_score", "transaction_id"], ascending=[False, True]
    ).reset_index(drop=True)


st.caption("PORTFOLIO PROJECT · SYNTHETIC BANKING DATA")
st.title("AML Transaction Monitoring")
st.write("Prioritize transactions for analyst review and measure the trade-off between coverage and workload.")

summary_path = DATA_DIR / "evaluation_summary.json"
required_paths = [summary_path, DATA_DIR / "validation_alerts.csv", DATA_DIR / "test_alerts.csv"]
missing_files = [path.name for path in required_paths if not path.exists()]
if missing_files:
    st.info("Generate the dashboard data by running fraud.ipynb from top to bottom first.")
    st.write("Missing files: " + ", ".join(missing_files))
    st.code("python -m jupyter lab fraud.ipynb", language="bash")
    st.stop()

try:
    summary = json.loads(summary_path.read_text())
    records = {row["Split"]: row for row in summary["splits"]}
    fixed_test = records["Test"]
    fixed_fraction = float(summary["review_fraction"])
    model_name = summary["model"]
except (ValueError, KeyError, TypeError, OSError) as exc:
    st.error(f"Cannot read the benchmark summary: {exc}. Rerun the notebook exports.")
    st.stop()

st.subheader(f"Held-out test benchmark · {fixed_fraction:.0%} review budget")
st.caption(
    f"{model_name}, selected using validation data only. "
    f"Test period: {fixed_test['Start']} to {fixed_test['End']}."
)
a, b, c, d = st.columns(4)
a.metric("Laundering recall", f"{fixed_test['Recall at Budget']:.1%}")
b.metric("Review precision", f"{fixed_test['Precision at Budget']:.2%}")
c.metric("Transactions reviewed", f"{fixed_test['Transactions Reviewed']:,}")
d.metric("False alerts", f"{fixed_test['False Alerts']:,}")
st.caption(
    f"Caught {fixed_test['Laundering Caught']:,} of {fixed_test['Labelled Laundering']:,} "
    f"labelled laundering transactions. Average precision: {fixed_test['Average Precision']:.4f}. "
    "This benchmark stays fixed when you change the exploratory controls below."
)

st.sidebar.header("Explore a review queue")
split = st.sidebar.selectbox("Dataset", ["Test", "Validation"])
path = DATA_DIR / f"{split.lower()}_alerts.csv"
try:
    alerts = load_alerts(str(path), path.stat().st_mtime_ns)
    record = records[split]
    fixed_budget = max(1, int(len(alerts) * fixed_fraction))
    if (
        len(alerts) != record["Transactions"]
        or int(alerts["Is Laundering"].sum()) != record["Labelled Laundering"]
        or int(alerts.head(fixed_budget)["Is Laundering"].sum()) != record["Laundering Caught"]
    ):
        raise ValueError("The alert export does not match the benchmark summary")
except (ValueError, KeyError, TypeError, OSError) as exc:
    st.error(f"Cannot load the {split.lower()} queue: {exc}. Rerun the notebook exports.")
    st.stop()

budget = int(st.sidebar.number_input(
    "Transactions to review",
    min_value=1,
    max_value=len(alerts),
    value=fixed_budget,
    step=1,
    key=f"budget_{split}",
))
st.sidebar.caption(
    f"Default: {fixed_fraction:.0%} of the full {split.lower()} period. "
    "Changing the budget explores the saved ranking; it does not retrain or select a model."
)
st.sidebar.info("Scores rank transactions. They are not calibrated laundering probabilities.")

selected = alerts.head(budget)
caught = int(selected["Is Laundering"].sum())
total_positive = int(alerts["Is Laundering"].sum())
precision = caught / budget
recall = caught / total_positive if total_positive else 0.0
prevalence = total_positive / len(alerts)

st.divider()
st.subheader(f"{split} queue · {budget:,} transactions selected")
st.caption(
    f"{record['Start']} to {record['End']} · {len(alerts):,} transactions · "
    f"{prevalence:.3%} labelled laundering. The review budget covers this entire period, not one day."
)
if budget != fixed_budget:
    st.info("Exploratory budget: use the fixed 1% benchmark above when reporting this project's test result.")
a, b, c, d = st.columns(4)
a.metric("Labelled laundering caught", f"{caught:,} / {total_positive:,}")
b.metric("Precision at this budget", f"{precision:.2%}")
c.metric("Recall at this budget", f"{recall:.1%}")
d.metric("False alerts at this budget", f"{budget - caught:,}")

queue_tab, budget_tab, notes_tab = st.tabs(["Alert queue", "Budget trade-offs", "Evaluation notes"])

with queue_tab:
    show_labels = st.checkbox("Show known labels for evaluation")
    columns = [
        "transaction_id", "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
        "Amount Paid", "Payment Currency", "Payment Format", "risk_score", "review_context",
    ]
    if show_labels:
        columns.append("Is Laundering")
    st.dataframe(
        selected[columns].head(500),
        width="stretch",
        hide_index=True,
        column_config={
            "transaction_id": st.column_config.NumberColumn("Transaction ID", format="%d"),
            "risk_score": st.column_config.ProgressColumn("Ranking score", min_value=0, max_value=1, format="%.4f"),
            "Amount Paid": st.column_config.NumberColumn("Amount paid", format="%.2f"),
            "review_context": st.column_config.TextColumn("Review context", width="large"),
        },
    )
    st.caption(f"Showing {min(budget, 500):,} selected alerts (up to 500). Equal scores are ordered by transaction ID, matching notebook evaluation.")
    download_rows = min(budget, 10_000)
    st.download_button(
        f"Download first {download_rows:,} selected alerts (CSV)",
        data=selected[columns].head(download_rows).to_csv(index=False),
        file_name=f"{split.lower()}_top_{download_rows}_alerts.csv",
        mime="text/csv",
    )

    st.subheader("Inspect a transaction")
    # A budget-specific key keeps an old rank from exceeding a reduced budget.
    rank = int(st.number_input("Alert rank", min_value=1, max_value=budget, value=1, key=f"rank_{split}_{budget}"))
    transaction = selected.iloc[rank - 1]
    left, right = st.columns(2)
    left.write(f"**Sender:** {transaction['From Bank']} / {transaction['Account']}")
    left.write(f"**Receiver:** {transaction['To Bank']} / {transaction['Account.1']}")
    left.write(f"**Payment format:** {transaction['Payment Format']}")
    right.write(f"**Amount:** {transaction['Amount Paid']:,.2f} {transaction['Payment Currency']}")
    right.write(f"**Timestamp:** {transaction['Timestamp']}")
    right.write(f"**Ranking score:** {transaction['risk_score']:.4f}")
    st.info(transaction["review_context"])
    if show_labels:
        st.write("**Known label:** " + ("Laundering" if transaction["Is Laundering"] else "Not laundering"))
    st.caption("Review context describes transaction attributes. It does not explain the model's exact reasoning or establish wrongdoing.")

with budget_tab:
    st.write("How much labelled laundering is captured as review capacity increases?")
    fractions = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    cumulative_caught = alerts["Is Laundering"].to_numpy().cumsum()
    curve_rows = []
    for fraction in fractions:
        count = max(1, int(len(alerts) * fraction))
        found = int(cumulative_caught[count - 1])
        curve_rows.append({
            "Review budget (%)": fraction * 100,
            "Transactions reviewed": count,
            "Recall (%)": 100 * found / total_positive if total_positive else 0.0,
            "Precision (%)": 100 * found / count,
            "False alerts": count - found,
        })
    curve = pd.DataFrame(curve_rows)
    st.line_chart(curve, x="Review budget (%)", y="Recall (%)")
    st.dataframe(curve.round(2), width="stretch", hide_index=True)
    st.caption("These are retrospective scenarios. Exploring test budgets does not change the preselected 1% reporting policy.")

with notes_tab:
    result_columns = [
        "Split", "Transactions", "Labelled Laundering", "Transactions Reviewed",
        "Laundering Caught", "False Alerts", "Precision at Budget", "Recall at Budget", "Average Precision",
    ]
    st.write("**Fixed-budget results**")
    result_table = pd.DataFrame(summary["splits"])[result_columns].copy()
    for column in ["Precision at Budget", "Recall at Budget"]:
        result_table[column] = result_table[column].map(lambda value: f"{value:.2%}")
    st.dataframe(result_table, width="stretch", hide_index=True)
    st.markdown("""
- **Selection:** compare logistic regression and random forest on validation recall at a 1% review budget; evaluate only the selected model on the later test period.
- **History:** sender features use strictly earlier transactions in the same currency, without their labels. Transactions sharing a timestamp cannot use each other's history.
- **Metrics:** precision measures useful alerts; recall measures labelled laundering found. Average precision summarizes ranking performance across thresholds.
- **Limitations:** synthetic data, changing prevalence over time, and a retrospective budget. This is a portfolio prototype, not a deployed detection system.
""")
    st.caption("Generated by fraud.ipynb. See README.md for reproducible setup, results, and lessons learned.")
