# Predicting Credit Card Fraud Transactions Using Machine Learning

## Project Overview
This project focuses on detecting **fraudulent credit card transactions** using **machine learning classification techniques**.

Credit card fraud is a major concern for financial institutions. It’s crucial that banks and card companies can identify fraudulent transactions in real time to prevent financial loss and protect customers.  
The goal of this project is to build a model that can accurately classify whether a given transaction is **fraudulent (1)** or **non-fraudulent (0)**.

---

## Dataset Information

### Context
The dataset contains transactions made by **European cardholders** in **September 2013**.  
It covers **two days of transactions**, totaling **284,807 records**, with **492 fraud cases**.

- **1 = Fraudulent Transaction**  
- **0 = Non-Fraudulent Transaction**

**Imbalance Warning:**  
Fraudulent cases represent only **0.172%** of all transactions — this makes the dataset **highly imbalanced**, which is common in fraud detection problems.

**Dataset Source:**  
[Kaggle: Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?select=creditcard.csv)

> Note: Feature names are anonymized for privacy and security reasons by the data provider.

---

## 🧠 Project Goal
The primary goal is to develop a **classification model** capable of detecting fraudulent transactions from historical data.

We aim to achieve:
- High **Recall** — correctly identifying most fraudulent transactions  
- Balanced **Precision** — minimizing false alarms  
- Good **F1-Score** — balancing recall and precision effectively  

---

## Proof of Concept (POC)
The project will be considered successful if it achieves: **Recall ≥ 90%**

**Why Recall?**  
In fraud detection, missing a fraud (false negative) is far more costly than a false alarm (false positive).  
A high recall ensures that most fraudulent transactions are caught.

---

## Key Evaluation Metrics

| Metric | Description | Why It Matters |
|--------|--------------|----------------|
| **Recall (Sensitivity)** | Measures how many actual fraud cases were correctly detected. | Critical for catching as many frauds as possible. |
| **Precision** | Measures how many of the predicted frauds were actually frauds. | Prevents flagging too many legitimate transactions. |
| **F1-Score** | Harmonic mean of Precision and Recall. | Balances both metrics for overall performance. |
| **Accuracy** | Percentage of total correct predictions. | Not useful alone in imbalanced data (can be misleading). |
| **AUC-ROC Curve** | Measures model’s ability to distinguish between classes. | Visual performance comparison across thresholds. |

---

## Challenges in Real-World Fraud Detection

### Noisy & Imperfect Data
Real-world data contains errors, missing values, and inconsistencies that can reduce model performance.

### Overlapping Class Distributions
Fraudulent and non-fraudulent transactions often have similar patterns, making perfect classification impossible.

### Concept Drift
Fraudsters evolve their strategies — models must be **retrained regularly** to adapt to new transaction behaviors.

---

## Steps Taken in the Project

1. **Data Loading & Cleaning** — Handle missing values and prepare data.  
2. **Exploratory Data Analysis (EDA)** — Analyze transaction trends and imbalance distribution.  
3. **Data Balancing** — Use techniques like **SMOTE** or **undersampling** to handle class imbalance.  
4. **Feature Scaling** — Normalize continuous features for algorithm compatibility.  
5. **Model Training** — Train classification models (e.g., Logistic Regression, Random Forest, XGBoost).  
6. **Model Evaluation** — Assess using Recall, Precision, F1, AUC-ROC.  
7. **Model Comparison & Tuning** — Optimize hyperparameters for better performance.  
8. **Deployment Preparation** — Save best-performing model and visualize results.  

---

## Tools Used

- **Python 3.x**
- **Pandas** — Data manipulation  
- **NumPy** — Numerical computation  
- **Scikit-Learn** — Machine learning algorithms  
- **Matplotlib / Seaborn** — Data visualization  
- **Imbalanced-Learn** — For oversampling techniques (e.g., SMOTE)  
- **Joblib** — For saving trained models  
- **Streamlit (optional)** — For deployment and interactive dashboard  

---

## Proof of Deployment (Future Work)
If the model meets the Proof of Concept (POC) target (Recall ≥ 90%), it will be deployed on:
- **Streamlit Community Cloud** for public interaction.
  

---

## Contact
For questions, feedback, or collaborations:  
📧 **nosakhareasowata94@gmail.com**

GitHub Repo: [Nosakhare-code/credit-card-fraud-detection](https://github.com/Nosakhare-code/credit-card-fraud-detection)

---

## 🧾 License
This project is released under the **MIT License** you are free to use, modify, and distribute it for educational or research purposes.

---

### Acknowledgements
Special thanks to:
- **Kaggle** for providing the dataset  
- **ULB Machine Learning Group** for the original research dataset  
- The open-source community for developing powerful Python libraries used in this project

