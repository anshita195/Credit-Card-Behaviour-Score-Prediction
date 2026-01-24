# Credit Card Default Prediction

## Project Overview
This project implements a machine learning pipeline aimed at predicting the probability of credit card default for the upcoming month. The system analyzes customer demographic data and a six-month window of historical financial activity to classify account holders into risk categories.

The primary objective is to assist financial institutions in reducing financial liability by accurately identifying high-risk customers. The solution prioritizes **Recall** and the **F2-score** as the decisive performance metrics, reflecting the business imperative that identifying potential defaulters (minimizing False Negatives) is critical for effective risk management.

## Dataset Description
The analysis utilizes a dataset comprising customer demographic details and repayment history. Key data attributes include:
- **Demographics:** Education, Sex, Marital Status, Age.
- **Financial History:** Credit limits, payment status (`pay_0` through `pay_6`), monthly bill statements, and previous payment amounts.
- **Target Variable:** `next_month_default` (Binary Classification: 1 indicates default, 0 indicates non-default).

*Note: The dataset exhibits a significant class imbalance, with a default rate of approximately 19%.*

## Methodology

### 1. Data Preprocessing
To ensure data integrity and model robustness, specific imputation strategies were applied:
- **Temporal Features (Payment Status):** A "fill-forward" approach was employed, assuming consistency in customer repayment behavior over time.
- **Quantitative Features:** Missing values in bill amounts and credit limits were imputed using the median to mitigate the influence of outliers.
- **Categorical Features:** Missing demographic attributes were imputed using the mode.

### 2. Feature Engineering
New features were derived to capture underlying patterns in financial behavior:
- **Payment Delay Statistics:** Calculation of Min, Max, Mean, and Standard Deviation of delay months to quantify risk severity.
- **Billing Stability:** Analysis of the average monthly change and standard deviation in bill and payment amounts.
- **Credit Utilization Ratio:** Derived as the ratio of the average bill amount to the credit limit, indicating the customer's dependency on available credit.

### 3. Class Imbalance Handling
Given the minority class prevalence (19%), the project employs **SMOTE (Synthetic Minority Over-sampling Technique)** to synthesize instances of the minority class in the training set. Additionally, inverse-frequency class weights were integrated into the loss functions of tree-based models to strictly penalize the misclassification of the positive class.

### 4. Model Selection
The following algorithms were implemented, trained, and evaluated:
- Logistic Regression
- Decision Trees
- Random Forest Classifier
- XGBoost
- LightGBM
- Majority-Vote Ensemble

### 5. Evaluation Metrics
The model optimization process prioritized the **F2-score** ($\beta=2$) over standard accuracy. This metric places higher weight on Recall than Precision, aligning with the specific business objective to capture the maximum number of true defaulters, even if it results in a higher False Positive rate.

## Experimental Results
The analysis indicated that recent payment delay indicators (`pay_0` to `pay_6`) account for the majority of the variance in default probability, whereas demographic features contributed only marginally to predictive power.

**Best Performing Model: Random Forest Classifier**
- **AUC Score:** 0.78
- **F2-Score:** 0.60 (achieved after threshold tuning at 0.28)

The Random Forest model demonstrated the highest discrimination capability among all tested algorithms and was selected as the final model for deployment.

## Tech Stack
- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM
- **Imbalance Handling:** Imbalanced-learn (SMOTE)
- **Model Interpretability:** SHAP

## Installation and Usage

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
2. **Install the required dependencies:**
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm imbalanced-learn shap
3. **Execution: The analysis is contained within the Jupyter Notebook. To run the pipeline:**
   jupyter notebook Credit_Card_Behaviour_Score_Prediction.ipynb
(Ensure that the dataset files (train_dataset_final1.csv and validate_dataset_final.csv) are located in the working directory.)

## Directory Structure
.
├── code_22123007_credit.ipynb   # Main source code for analysis and modeling
├── report_22123007_credit.pdf   # Detailed project report and findings
├── output/                      # Generated visualizations and metrics
└── README.md                    # Project documentation
