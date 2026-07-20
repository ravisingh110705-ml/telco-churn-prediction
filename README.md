# Telco Customer Churn Prediction

A binary classification project predicting whether a telecom customer will churn (cancel their service), with a focus on the real-world tradeoff between accuracy and recall when the goal is catching at-risk customers.

## Problem

Telecom companies lose revenue every time a customer cancels. Being able to flag at-risk customers *before* they leave lets a retention team step in with offers or outreach. This project builds and compares several models to predict churn from a customer's account details (contract type, tenure, billing info, subscribed services) and explores which model is actually most useful for that real business goal — not just which one has the highest accuracy.

## Dataset

- 7,043 customers, 21 original columns (IBM Telco Customer Churn dataset)
- Target: `Churn` (Yes/No)
- Mix of numeric (tenure, MonthlyCharges, TotalCharges), binary (gender, Partner, PhoneService), and multi-category features (Contract, InternetService, PaymentMethod, and several service add-ons)

## Data Cleaning Highlights

- **`TotalCharges` stored as text with hidden blank spaces** — found ~11 rows with `" "` instead of a number. Traced these back to customers with `tenure = 0` (brand new customers who genuinely hadn't been billed yet), so these were filled with `0` rather than the median — the logically correct value given their situation, not just a statistical placeholder.
- **`customerID` dropped** — pure identifier, no predictive value.
- **Encoding decisions made deliberately per column type:**
  - Binary (gender, Partner, Dependents, PhoneService, PaperlessBilling) → mapped to 0/1
  - Nominal, no order (InternetService, PaymentMethod, and service add-ons) → one-hot encoded
  - `Contract` → manually ordinal-mapped (Month-to-month=0, One year=1, Two year=2), since it has a genuine real-world order that `LabelEncoder` couldn't be trusted to preserve correctly

## Correlation Check (before modeling)

Learned from an earlier project that correlation should be checked *before* investing time in models. Strongest relationships with Churn:
- `Contract`: -0.397 (longer contracts → less churn)
- `tenure`: -0.352 (longer-tenured customers → less churn)
- `InternetService_Fiber optic`: +0.308
- `PaymentMethod` (Electronic check): +0.302

All consistent with real-world churn behavior — confirmed this was a genuinely learnable dataset before training anything.

## Models Compared

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | **0.821** | 0.685 | 0.601 | 0.640 |
| SVM | 0.813 | 0.695 | 0.525 | 0.598 |
| XGBoost | 0.798 | 0.639 | 0.542 | 0.586 |
| Random Forest | 0.798 | 0.661 | 0.485 | 0.560 |
| Decision Tree | 0.797 | 0.657 | 0.488 | 0.560 |
| KNN | 0.770 | 0.576 | 0.496 | 0.533 |
| Naive Bayes | 0.665 | 0.435 | 0.890 | 0.585 |

## The Real Finding: Accuracy vs. Recall

Logistic Regression wins on raw accuracy — but in a churn context, a **false negative** (predicting a customer will stay when they're actually about to leave) is usually more costly than a false alarm. Naive Bayes, despite the worst accuracy, catches **89% of actual churners** vs. Logistic Regression's 60%.

To find a better middle ground, I retrained every model with `class_weight='balanced'` (and `scale_pos_weight` for XGBoost, which uses a different parameter):

| Model | Accuracy (orig → balanced) | Recall (orig → balanced) |
|---|---|---|
| Logistic Regression | 0.821 → 0.749 | 0.601 → 0.823 |
| SVM | 0.813 → 0.754 | 0.525 → 0.804 |
| Random Forest | 0.798 → 0.784 | 0.485 → 0.649 |
| Decision Tree | 0.797 → 0.763 | 0.488 → 0.772 |
| XGBoost | 0.798 → 0.770 | 0.542 → 0.681 |

**Balancing consistently trades some accuracy for a large recall gain across every model** — a real, repeatable pattern, not a fluke of one algorithm. Given the business cost of missing a churner, **SVM (balanced)** was chosen as the final model: a strong recall improvement (0.804) while keeping a smaller accuracy cost than most alternatives.

## Final Model

- **SVM with `class_weight='balanced'`**
- Test Accuracy: 0.752
- Test Recall: 0.812
- Bundled into a single `sklearn` Pipeline (scaling, encoding, and the classifier) so it can be reused on raw, unprocessed customer data without manually repeating any cleaning steps.

## Files

- `main.ipynb` — full exploratory analysis: cleaning decisions, EDA, the 7-model comparison, and the class-imbalance experiments
- `train_pipeline.ipynb` — final, clean training script producing the saved pipeline
- `telco_pipeline.pkl` — the saved, trained pipeline (preprocessing + model bundled together)
- `data/Telco-Customer-Churn.csv` — raw dataset

## Tech Stack

Python, pandas, scikit-learn, XGBoost, joblib

## Note on a Related Failed Experiment

While practicing regression around the same time, I worked with a house price dataset where correlation checks revealed the price column had no real relationship with any feature (max correlation ~0.05), later confirmed by both Linear Regression and Random Forest scoring negative R². Rather than force a model onto data with no genuine signal, I diagnosed the issue and moved to a properly structured housing dataset instead. Not every dataset is worth modeling — recognizing that early is part of the process.

## Other Projects
- [Titanic Survival Prediction](https://github.com/ravisingh110705-ml/titanic-survival-prediction)
- [Heart Disease Prediction](https://github.com/ravisingh110705-ml/heart-disease-prediction)
- [Medical Insurance Cost Prediction](https://github.com/ravisingh110705-ml/medical-insurance-cost-prediction) 
