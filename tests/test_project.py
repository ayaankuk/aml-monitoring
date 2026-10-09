"""Regression checks for chronological features, evaluation, and dashboard controls."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "fraud.ipynb").read_text())


def cell_source(cell_id):
    return "".join(next(cell for cell in NOTEBOOK["cells"] if cell["id"] == cell_id)["source"])


def budget_evaluator():
    """Load just the notebook function, without executing training cells."""
    tree = ast.parse(cell_source("d43e8b13"))
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = {"np": np, "average_precision_score": average_precision_score}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<notebook evaluator>", "exec"), namespace)
    return namespace["evaluate_at_budget"]


class EvaluationTests(unittest.TestCase):
    def test_history_excludes_current_timestamp_future_and_other_currencies(self):
        frame = pd.DataFrame({
            "Timestamp": pd.to_datetime([
                "2022-01-01", "2022-01-02", "2022-01-02", "2022-01-03", "2022-01-03",
            ]),
            "From Bank": [1] * 5,
            "To Bank": [2] * 5,
            "Account": ["A"] * 5,
            "Payment Currency": ["USD", "USD", "USD", "USD", "EUR"],
            "Amount Paid": [10.0, 20.0, 40.0, 30.0, 100.0],
        })
        namespace = {
            "df": frame, "pd": pd,
            "train_cutoff": pd.Timestamp("2022-01-02"),
            "test_cutoff": pd.Timestamp("2022-01-03"),
        }
        exec(cell_source("0bf05f0e"), namespace)
        result = namespace["df"]
        self.assertEqual(result.previous_transaction_count.tolist(), [0, 1, 1, 3, 0])
        np.testing.assert_allclose(
            result.previous_average_amount, [np.nan, 10, 10, 70 / 3, np.nan], equal_nan=True,
        )

    def test_budget_counts_and_ties(self):
        evaluate = budget_evaluator()
        # The tied score must select row 0 before row 1; row 2 ranks first.
        result = evaluate([1, 0, 1, 0], [0.8, 0.8, 0.9, 0.1], 0.5)
        self.assertEqual(result["Transactions Reviewed"], 2)
        self.assertEqual(result["Laundering Caught"], 2)
        self.assertEqual(result["False Alerts"], 0)
        self.assertEqual(result["Precision at Budget"], 1)
        self.assertEqual(result["Recall at Budget"], 1)
        self.assertEqual(result["Lift over Random"], 2)
        small = evaluate([0, 1], [0.9, 0.1], 0.01)
        self.assertEqual(small["Transactions Reviewed"], 1)
        self.assertEqual(small["Laundering Missed"], 1)
        no_positives = evaluate([0, 0], [0.9, 0.1], 0.5)
        self.assertTrue(np.isnan(no_positives["Recall at Budget"]))

    def test_exports_reproduce_frozen_benchmark(self):
        summary_path = ROOT / "app/evaluation_summary.json"
        if not summary_path.exists() or not (ROOT / "app/test_alerts.csv").exists():
            self.skipTest("Run fraud.ipynb first to generate the evaluation artifacts")
        summary = json.loads(summary_path.read_text())
        candidates = summary["validation_models"]
        winner = max(candidates, key=lambda row: row["Recall at Budget"])["Model"]
        self.assertEqual(summary["model"], winner)
        for record in summary["splits"]:
            with self.subTest(split=record["Split"]):
                queue = pd.read_csv(
                    ROOT / f"app/{record['Split'].lower()}_alerts.csv",
                    usecols=["transaction_id", "risk_score", "Is Laundering"],
                    float_precision="round_trip",
                )
                self.assertFalse(queue.transaction_id.duplicated().any())
                # Shuffle before sorting to prove score ties have a shared ID-based order.
                ranked = queue.sample(frac=1, random_state=42).sort_values(
                    ["risk_score", "transaction_id"], ascending=[False, True],
                )
                self.assertEqual(queue.transaction_id.tolist(), ranked.transaction_id.tolist())
                result = budget_evaluator()(
                    ranked["Is Laundering"], ranked["risk_score"], summary["review_fraction"],
                )
                for metric, value in result.items():
                    self.assertAlmostEqual(value, record[metric], places=10, msg=metric)


class DashboardTests(unittest.TestCase):
    def test_missing_exports_show_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "dashboard.py"
            script.write_text((ROOT / "app/dashboard.py").read_text())
            app = AppTest.from_file(str(script)).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn("running fraud.ipynb", app.info[0].value)

    def test_budgets_labels_and_dataset_switch_preserve_benchmark(self):
        if not (ROOT / "app/test_alerts.csv").exists():
            self.skipTest("Run fraud.ipynb first to generate the dashboard data")
        summary = json.loads((ROOT / "app/evaluation_summary.json").read_text())
        test = next(row for row in summary["splits"] if row["Split"] == "Test")
        app = AppTest.from_file(str(ROOT / "app/dashboard.py")).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        original_benchmark = [metric.value for metric in app.metric[:4]]
        self.assertEqual(app.number_input(key="budget_Test").value, test["Transactions Reviewed"])
        self.assertNotIn("Is Laundering", app.dataframe[0].value.columns)
        app.checkbox[0].check().run(timeout=60)
        self.assertIn("Is Laundering", app.dataframe[0].value.columns)
        app.number_input(key="budget_Test").set_value(1).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe[0].value), 1)
        self.assertEqual(app.number_input(key="rank_Test_1").value, 1)
        self.assertEqual([metric.value for metric in app.metric[:4]], original_benchmark)
        app.selectbox[0].select("Validation").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual([metric.value for metric in app.metric[:4]], original_benchmark)
        validation = next(row for row in summary["splits"] if row["Split"] == "Validation")
        self.assertEqual(app.number_input(key="budget_Validation").value, validation["Transactions Reviewed"])


if __name__ == "__main__":
    unittest.main()
