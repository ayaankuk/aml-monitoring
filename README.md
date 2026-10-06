# AML Transaction Monitoring with Machine Learning

A machine learning project that identifies potentially suspicious transactions and prioritizes them for review.

This project explores an important challenge in anti-money laundering (AML): detecting laundering activity while keeping the number of false alerts manageable.

## Project Overview

I built an AML transaction-monitoring prototype that prioritizes transactions for investigation. I engineered features from account history, compared logistic regression and random forest using a time-based split, and evaluated detection against false alerts and review capacity. I then built a dashboard so users could explore how much labelled laundering the model catches at different investigation budgets.

## Workflow

1. **Explore and clean the data**
   - Check missing values, duplicate records, and data types.
   - Examine transaction amounts and the distribution of laundering labels.
   - Investigate unusual values before deciding how to handle them.

2. **Prepare model inputs**
   - Select relevant transaction features.
   - Encode categorical variables as needed.
   - Separate the target label from the input features.
   - Split the data into training and validation sets.

3. **Train and compare models**
   - Train Logistic Regression and Random Forest.
   - Compare their ability to detect laundering transactions.
   - Examine the trade-off between missed cases and false alerts.

4. **Generate alerts**
   - Score validation transactions using the selected model.
   - Apply a decision threshold to flag suspicious transactions.
   - Export results for review in the dashboard.

5. **Review results**
   - Inspect flagged transactions.
   - Explore how the alert threshold affects detection and review workload.

## Dataset

- **Source:** HI-Small_Trans.csv --> https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data?select=HI-Small_Trans.csv
- **Target:** is_laundering
- **Dataset type:** synthetic

Raw data is excluded from this repository. Download it from the original source and follow its licensing terms.

## Model Evaluation

Laundering transactions represent a small share of the dataset, so accuracy alone can give a misleading picture of performance.

The comparison focuses on:

| Metric | What it measures |
|--------|-----------------|
| Precision | The share of flagged transactions that are labelled as laundering |
| Recall | The share of labelled laundering transactions detected |
| F1 score | The balance between precision and recall |
| Average precision | Performance across precision-recall thresholds |
| Alert volume | The number of transactions sent for review |


## Running the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/ayaankukreja/aml-monitoring.git
   cd aml-monitoring
