# SYRIATEL CUSTOMER CHURN PREDICTION

![Kastomaaa](https://media.istockphoto.com/id/530538787/photo/customer-engagement.jpg?s=1024x1024&w=is&k=20&c=dztp7HBDXxSlszUrC63IWjI4MY0YJqfdA4U7-AbNamg=)


## Executive Summary

Customer churn is one of the most significant challenges faced by telecommunications companies. When customers leave, the business loses recurring revenue and must spend more to acquire new clients. This project uses data-driven analysis and machine learning to help SyriaTel identify customers who are most likely to leave, enabling the business to intervene early and improve retention.

## Project Purpose

The goal of this project is to build a reliable system that can:

* Identify customers who are at risk of leaving.
* Reveal the factors that most influence churn.
* Support business teams with insights they can act upon.
* Provide a deployable model that can be integrated into business operations.

This work blends analytics and machine learning into a practical business-focused solution.

## Data Overview

The analysis uses a dataset of 3,333 SyriaTel customers, including:

* Usage patterns (day, evening, night, international calls).
* Customer service activity.
* Plan subscriptions (international plan, voicemail plan).
* Customer demographic information.
* Churn status (whether the customer left the company).

There were **no missing values** and all records were complete. Additional helpful variables were created, such as total usage, service intensity, and usage ratios.

## Key Insights from Analysis

### 1. Customer Behaviour Patterns

* Customers who frequently call customer service show a much higher likelihood of leaving.
* Those on international plans churn significantly more.
* Heavy users across day, evening, and night periods fall into clear behavioral groups, useful for segmentation.

### 2. Churn Rate

* Only about **15% of customers churned**, making this a *highly imbalanced* problem.
* Special techniques were required to ensure the model learns from the small churn group.

### 3. Influential Factors

Across multiple models, the following variables consistently emerged as the strongest indicators of churn:

* Number of customer service calls.
* Whether the customer has an international plan.
* Voicemail plan subscription.
* Patterns of day and night usage.
* Customer’s state.

## Machine Learning Approach

The modelling was executed in three phases:

### **1. Baseline Models**

Several models were trained using default settings to establish initial performance benchmarks:

* Logistic Regression
* Random Forest
* XGBoost
* CatBoost

### **2. Addressing Class Imbalance**

Since churners represent only a small portion of the dataset, two strategies were used:

* **Class weighting:** giving the model stronger focus on the minority class.
* **SMOTE:** a technique that creates synthetic churn examples to balance the dataset.

### **3. Model Optimization**

The best-performing models were then improved using:

* RandomizedSearchCV
* GridSearchCV
* Optuna (advanced optimization)

This ensured that the final model was accurate, stable, and ready for deployment.

## Model Performance

Across all iterations, **XGBoost** and **CatBoost** consistently delivered the best results. Their strengths include:

* High ability to correctly identify churners.
* Strong overall accuracy.
* Excellent balance between false positives and false negatives.
* Better generalization to new, unseen data.

The final selected model achieved:

* **High accuracy (above 94%)**
* **Strong recall**, ensuring most churners are correctly detected
* **Excellent ROC–AUC scores**, indicating reliable predictive power

## Deployment Readiness

The final model is saved using joblib and packaged inside a pipeline that:

* Preprocesses new customer data
* Applies the trained model
* Outputs a churn prediction instantly

This makes the system easy to integrate into customer management platforms or business dashboards.

## Business Value

This solution enables SyriaTel to:

* Detect at-risk customers early.
* Prioritize retention campaigns.
* Reduce revenue loss.
* Improve customer satisfaction.
* Equip customer experience, marketing, and executive teams with actionable insights.

With the ability to pinpoint high-risk customers, SyriaTel can strategically allocate retention resources where they matter most.

## Tools and Technologies

* Python
* Pandas and NumPy
* Scikit-learn
* Ensemble Methods
* Optuna
* Matplotlib & Seaborn
* Imbalanced-learn
* GridSearchCV and RandomizedSearchCV

## Author

**Norman Kodi**
Machine Learning & Data Science Practitioner

For inquiries or collaboration, feel free to connect.