# AML Transaction Monitoring with Machine Learning

A machine learning prototype that ranks synthetic banking transactions for analyst review under a fixed **1% review budget**. The project combines chronological evaluation, sender-history features, logistic regression and random forest baselines, and an interactive Streamlit dashboard.

**Held-out test result:** the selected random forest caught **812 of 1,561 labelled laundering transactions (52.02% recall)** while reviewing **7,620 of 762,099 transactions**. Precision was **10.66%**, with **6,808 false alerts**. These are synthetic-data results from a retrospective review simulation.

## The problem

An analyst cannot review every transaction. The objective is to rank transactions so a limited review queue captures as much labelled laundering as possible. Accuracy alone would hide the challenge: only about 0.10% of validation transactions and 0.20% of test transactions are labelled laundering.

The primary metric is **recall at a 1% review budget**, with precision, false alerts, and average precision reported alongside it. The budget is `max(1, floor(0.01 × number of transactions))` for each complete evaluation period. It is not a daily review limit or a 0.50 score threshold.

## Workflow

1. Check missing values and remove exact duplicate transactions.
2. Create chronological training, validation, and test splits, then engineer transaction and sender-history features.
3. Train logistic regression and random forest; select the model using validation recall at a 1% review budget.
4. Evaluate the selected model on the held-out test period and export ranked validation/test queues.
5. Inspect transactions and explore review-capacity trade-offs in the dashboard.

## Data and approach

The project uses `HI-Small_Trans.csv` from [IBM Transactions for Anti Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml). The dataset is synthetic; its research context is described in [IBM's dataset paper](https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models).

The notebook reads **5,078,345 transactions** and removes 9 exact duplicate rows, leaving **5,078,336**. It then sorts by timestamp and creates approximately 70% / 15% / 15% chronological splits without dividing transactions at the same timestamp across splits.

| Split | Period | Transactions | Labelled laundering | Prevalence |
| --- | --- | ---: | ---: | ---: |
| Training | Sep 1, 2022 00:00 – Sep 7, 2022 14:54 | 3,554,647 | 2,856 | 0.0803% |
| Validation | Sep 7, 2022 14:55 – Sep 9, 2022 03:15 | 761,590 | 760 | 0.0998% |
| Test | Sep 9, 2022 03:16 – Sep 18, 2022 16:18 | 762,099 | 1,561 | 0.2048% |

Features include hour, weekday, weekend status, transfers between different banks, log transaction amount, payment currency and format, and sender-history statistics. Sender history is grouped by bank, account, and currency. Counts and average amounts use only transactions **strictly before the current timestamp**, so transactions sharing a timestamp do not use each other's information.

The pipeline fits imputation, scaling, and categorical encoding on training data only. Both models use class weighting. The random forest uses 50 trees, depth 10, a minimum of 20 samples per leaf, and 100,000 bootstrap samples per tree to keep training practical.

Model selection uses validation recall at the fixed 1% review budget. The selected, training-only model is then scored on the later test period without refitting or tuning against test labels. Equal scores are ordered by the notebook's transaction ID, and the dashboard uses that same ordering.

## Results

### Validation model comparison

Both models review the same 7,615 transactions by count, taking their respective highest scores.

| Model | Laundering caught | Precision at 1% | Recall at 1% | Average precision |
| --- | ---: | ---: | ---: | ---: |
| Logistic regression | 16 / 760 | 0.21% | 2.11% | 0.0068 |
| Random forest | 384 / 760 | 5.04% | 50.53% | 0.0925 |

### Selected random forest: fixed-budget evaluation

| Split | Reviewed | Laundering caught | Precision | Recall | False alerts | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 7,615 | 384 / 760 | 5.04% | 50.53% | 7,231 | 0.0925 |
| Held-out test | 7,620 | 812 / 1,561 | 10.66% | 52.02% | 6,808 | 0.2314 |

Test precision is about **52× the test prevalence**, but **89.34% of reviewed transactions are still false alerts**, and 749 labelled laundering transactions are missed. Higher test precision should also be read alongside the increased test prevalence; it does not establish that the model improved over time.

The notebook retains a 0.50 threshold comparison as a secondary diagnostic. It is not the model-selection policy or the headline result. Exact metrics, split boundaries, and library versions are saved in [`app/evaluation_summary.json`](app/evaluation_summary.json).

### Reading the metrics

| Metric | What it measures |
| --- | --- |
| Review precision | The share of transactions selected for review that are labelled laundering |
| Laundering recall | The share of all labelled laundering transactions captured in the review queue |
| Transactions reviewed | The number of highest-ranked transactions selected for simulated review |
| False alerts | Selected transactions labelled as not laundering |
| Average precision | Ranking performance summarized across precision-recall thresholds |
| F1 score | The harmonic mean of precision and recall; included in the secondary threshold comparison |

## Run locally

Tested with **Python 3.12.5**. First clone the repository if you do not already have a local copy:

```bash
git clone https://github.com/ayaankuk/aml-monitoring.git
cd aml-monitoring
```

Use Python 3.12 and run the following commands from the repository root.

### 1. Create an environment and install dependencies

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --prefix .venv --name aml-monitoring --display-name "Python (AML Monitoring)"
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell and use your Python 3.12 executable to create the environment.

### 2. Download the data

Download the dataset from the Kaggle link above and extract **`HI-Small_Trans.csv`** into the repository root, beside `fraud.ipynb`. The CSV and generated alert queues are excluded from Git. A Kaggle sign-in may be required; follow the dataset's listed usage terms.

This is a roughly 476 MB input file containing over 5 million rows. The notebook holds multiple dataframes in memory, so allow several GB of available RAM. Runtime and memory use depend on the machine.

### 3. Run the notebook

```bash
python -m jupyter lab fraud.ipynb
```

Select the **Python (AML Monitoring)** kernel, then use **Restart Kernel and Run All Cells**. Run from the repository root so the relative dataset path resolves correctly.

For a non-interactive run:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=aml-monitoring --ExecutePreprocessor.timeout=1800 fraud.ipynb
```

A full run trains both models, selects the winner using validation, evaluates it on test, saves notebook outputs, and generates:

- `app/validation_alerts.csv` — the ranked validation queue.
- `app/test_alerts.csv` — the ranked test queue.
- `app/evaluation_summary.json` — the fixed-budget results and run metadata.

### 4. Start the dashboard

```bash
python -m streamlit run app/dashboard.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The dashboard includes a fixed held-out test benchmark, validation/test queue selection, an adjustable review count, a ranked queue, transaction inspection, optional known labels, CSV downloads, and budget trade-off charts. Changing the exploratory budget leaves the fixed 1% benchmark unchanged. The table displays up to 500 alerts, and downloads include up to 10,000 selected alerts.

If dashboard exports are missing, the app displays notebook-run instructions. Rerun the notebook after changing features or model settings so the exports and summary stay consistent. The dashboard reads saved predictions; it does not train or score new transactions.

### 5. Run the checks

```bash
python -m unittest discover -s tests -v
```

Checks cover historical-feature boundaries, score ties at the review-budget boundary, exported metrics, and dashboard behavior when changing budgets, toggling labels, switching datasets, or starting without exports. Artifact-dependent checks skip until the notebook exports exist.

## What I learned

- **Evaluation needs to reflect the workflow.** A review budget turns model scores into an analyst workload. Reporting recall alone hides the number of false alerts that someone must inspect.
- **Class imbalance changes what good performance means.** Precision, recall, average precision, and lift over prevalence are more informative for this project than headline accuracy. Class weighting alone does not guarantee a useful top-ranked queue, as the logistic regression comparison shows.
- **Historical features require careful time boundaries.** A sender's past behavior can provide context, but current or future transactions must not enter their historical aggregates. Currency-specific grouping also avoids averaging amounts in incompatible units.
- **Validation and test have different jobs.** Validation supports model selection; the later test period provides a separate estimate after those decisions are fixed. Preprocessing must be fitted on training data.
- **Reproducibility includes ranking details.** Equal model scores need a consistent tie-breaker. Otherwise the notebook and dashboard can show different results at the same budget.
- **A score and an explanation are different things.** The dashboard treats scores as rankings, and its review context describes transaction attributes rather than claiming to explain the model's exact reasoning.

## Limitations and next steps

This is a portfolio prototype built on synthetic data, not evidence of production AML performance. Labels are available for retrospective evaluation; they would not be known for a live review queue. Scores are not calibrated laundering probabilities, and a flagged transaction does not establish wrongdoing.

The split periods have different durations and different positive rates. Accounts may recur across splits, so the test measures performance on later transactions, not exclusively on unseen accounts. Historical features assume earlier transaction records remain available as time advances, including earlier records within the validation and test periods, without using their labels.

Future work could add rolling time-based validation, daily review budgets, a comparison with and without sender-history features, and receiver/network features. The existing test results should remain a recorded benchmark; further tuning after examining them would need a new untouched holdout for a fresh final estimate.

## Project files

```text
fraud.ipynb                    Data preparation, training, evaluation, exports
requirements.txt               Tested Python dependencies
app/dashboard.py               Streamlit review dashboard
app/evaluation_summary.json    Generated benchmark metrics and metadata
app/validation_alerts.csv       Generated locally; ignored by Git
app/test_alerts.csv             Generated locally; ignored by Git
tests/test_project.py           Evaluation and dashboard regression checks
```
