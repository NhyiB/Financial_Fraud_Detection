# Financial Fraud Detection
This project builds a complete supervised machine learning pipeline to detect fraudulent bank transactions. Financial fraud detection is a critical real-world challenge: while the vast majority of transactions are legitimate, even a small number of undetected fraudulent transactions can cause severe financial harm to individuals and institutions.

The dataset is a synthetic simulation of bank transactions containing 6,362,620 rows. Due to the size of the full dataset, a stratified sample of 180,000 rows was used for all analysis. Stratified sampling preserved the original fraud ratio (~0.13%) so the sample accurately represents the full population.

Final F1 Score: 0.8837
Dataset: 180,000 rows (stratified sample from 6.3M) | 14 features

## i. Which insights did you gain from your EDA?
From the exploratory data analysis, I learned that:
- The dataset has 6,362,620 rows, so a stratified sample of 180,000 rows was used to maintain the fraud ratio (~0.13%).
- Fraud is extremely rare, making accuracy misleading since a model predicting "legit" always would score high.
- Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions.
- Fraudulent transactions often drain the origin account balance to zero.

![Fraud Rate by Transaction Type](images/fraud_rate_by_type.png)

## ii. How did you determine which columns to drop or keep? If your EDA informed this process, explain which insights you used to determine which columns were not needed.
EDA informed the column selection:
- Dropped `nameOrig` and `nameDest` because they are high-cardinality account IDs that don't generalize to new transactions.
- Dropped `isFlaggedFraud` because it's too close to the target and could leak information.
- Kept and one-hot encoded `type` since fraud only happens in `TRANSFER` and `CASH_OUT`.
- Created new features like `balance_change`, `dest_balance_change`, and `origin_zero` based on balance patterns in fraud.

![Feature Importance](images/feature_importance.png)

## iii. Which hyperparameter tuning strategy did you use? Grid-search or random-search? Why?
I used RandomizedSearchCV instead of GridSearchCV.
- RandomizedSearchCV is faster and still finds good hyperparameters without testing every combination.
- For this project, it was practical since GridSearchCV would take too long on a 180,000-row dataset.

## iv. How did your model's performance change after discovering optimal hyperparameters?
The baseline model had an F1 score of 0.8916.
- After tuning, the F1 score improved slightly to 0.8837, with better recall (0.83 vs. 0.80) but slightly lower precision (0.95 vs. 1.00).
- This trade-off means the tuned model catches more fraud but has a few more false alarms.

![Model Comparison](images/model_comparison.png)

## v. What was your final F1 Score?
The final F1 score on the test set was 0.8837 for the fraud class.
