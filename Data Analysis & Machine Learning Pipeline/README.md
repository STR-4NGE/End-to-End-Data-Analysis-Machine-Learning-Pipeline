# Customer Churn Prediction: Portfolio Project

## 📊 Project Overview

A comprehensive end-to-end data science project demonstrating statistical analysis and machine learning techniques to predict customer churn. This project showcases core data science competencies including exploratory data analysis, statistical hypothesis testing, feature engineering, and predictive modeling.

**Author:** Data Science Portfolio  
**Date:** February 2026  
**Language:** Python 3.x

---

## 🎯 Project Objectives

1. Perform comprehensive exploratory data analysis on customer data
2. Conduct statistical hypothesis testing to identify churn drivers
3. Engineer meaningful features from raw data
4. Build and compare multiple machine learning models
5. Provide actionable business recommendations based on findings

---

## 📁 Project Structure

```
customer-churn-analysis/
│
├── customer_churn_analysis.py       # Main analysis script (Jupyter-ready)
├── README.md                         # Project documentation
│
├── outputs/
│   ├── eda_visualizations.png       # EDA plots and distributions
│   ├── correlation_heatmap.png      # Feature correlation matrix
│   ├── model_evaluation.png         # Model performance visualizations
│   ├── feature_importance.png       # Top predictive features
│   └── model_comparison.csv         # Model metrics comparison table
```

---

## 🔧 Technologies & Libraries

- **Data Manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Statistical Analysis:** scipy.stats
- **Machine Learning:** scikit-learn
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Metrics:** accuracy, precision, recall, F1-score, ROC-AUC

---

## 📈 Dataset Description

**Synthetic Customer Churn Dataset**

- **Size:** 5,000 customers
- **Features:** 13 predictor variables + 1 target variable
- **Class Distribution:** ~24% churn rate (realistic imbalance)

### Features:

| Feature | Type | Description |
|---------|------|-------------|
| Age | Numeric | Customer age (18-70) |
| Gender | Categorical | Male/Female |
| Tenure | Numeric | Months with company |
| MonthlyCharges | Numeric | Monthly subscription cost ($) |
| TotalCharges | Numeric | Total amount charged |
| Contract | Categorical | Month-to-month/One year/Two year |
| InternetService | Categorical | DSL/Fiber optic/No |
| OnlineSecurity | Categorical | Yes/No/No internet service |
| TechSupport | Categorical | Yes/No/No internet service |
| PaymentMethod | Categorical | Payment type |
| NumServices | Numeric | Number of services subscribed |
| CustomerServiceCalls | Numeric | Number of support calls |
| Churn | Binary | Target variable (0=No, 1=Yes) |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution analysis of all features
- Churn rate by categorical variables
- Numerical feature relationships
- Missing value identification and treatment

### 2. Statistical Hypothesis Testing

**Test 1: Monthly Charges (Mann-Whitney U)**
- **H₀:** Mean monthly charges are equal for churned and non-churned customers
- **Result:** Rejected (p < 0.001) - Significant difference exists

**Test 2: Tenure (Independent t-test)**
- **H₀:** Mean tenure is equal for churned and non-churned customers  
- **Result:** Rejected (p < 0.001) - Churned customers have lower tenure

**Test 3: Contract Type (Chi-square)**
- **H₀:** Contract type and churn are independent
- **Result:** Rejected (p < 0.001) - Strong association exists

### 3. Data Preprocessing

- Missing value imputation (median strategy)
- Feature engineering:
  - ChargesPerMonth: Average monthly cost
  - IsNewCustomer: Binary flag for tenure ≤ 6 months
  - HighValueCustomer: Above-median monthly charges
  - ServiceIntensity: Normalized service count
- One-hot encoding for categorical variables
- Standard scaling for numerical features

### 4. Machine Learning Models

Three models trained and evaluated:

1. **Logistic Regression** ⭐ (Best Performer)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

### 5. Model Evaluation

Comprehensive evaluation using:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Confusion matrices
- 5-fold cross-validation
- Feature importance analysis

---

## 📊 Key Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **0.763** | **0.515** | **0.290** | **0.371** | **0.775** |
| Random Forest | 0.761 | 0.519 | 0.112 | 0.184 | 0.747 |
| Gradient Boosting | 0.746 | 0.445 | 0.220 | 0.294 | 0.749 |

**Winner:** Logistic Regression (Highest ROC-AUC and balanced metrics)

### Top 5 Predictive Features

1. Contract Type (One year)
2. Total Charges
3. Charges Per Month
4. Monthly Charges
5. Tenure

---

## 💡 Business Insights

### Key Findings:

1. **Contract Flexibility is Critical**  
   Month-to-month contracts have 2.5x higher churn than annual contracts

2. **New Customers are at Risk**  
   Customers with tenure < 6 months show significantly higher churn probability

3. **Service Quality Matters**  
   Customers with >3 service calls are 40% more likely to churn

4. **Pricing Sensitivity**  
   Higher monthly charges correlate with increased churn risk

### Actionable Recommendations:

1. **Retention Programs:** Target customers in first 6 months with onboarding support
2. **Contract Incentives:** Offer discounts for annual contract upgrades
3. **Service Improvement:** Reduce need for customer service calls through better UX
4. **Pricing Strategy:** Review value proposition for high-charge customers
5. **Predictive Intervention:** Deploy model to identify at-risk customers proactively

---

## 🚀 How to Run

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### Execution

**Option 1: Jupyter Notebook**
```bash
# Convert to notebook format
jupyter nbconvert --to notebook --execute customer_churn_analysis.py

# Or copy code into Jupyter cells
```

**Option 2: Python Script**
```bash
python customer_churn_analysis.py
```

**Option 3: Interactive Python**
```bash
python -i customer_churn_analysis.py
```

### Expected Runtime
- Typical execution: 10-15 seconds
- Generates 4 PNG visualizations + 1 CSV file

---

## 📸 Visualizations

### 1. EDA Visualizations
- Churn distribution
- Age and tenure distributions by churn status
- Monthly charges boxplots
- Contract type vs churn rate
- Customer service calls impact

### 2. Correlation Heatmap
- Feature correlation matrix
- Relationship strengths visualization

### 3. Model Evaluation
- Confusion matrices for all three models
- ROC curves with AUC scores
- Side-by-side performance comparison

### 4. Feature Importance
- Random Forest feature importances
- Gradient Boosting feature importances
- Top 15 most influential variables

---

## 🎓 Skills Demonstrated

### Statistical Analysis
- ✅ Hypothesis testing (parametric & non-parametric)
- ✅ Chi-square test for categorical associations
- ✅ Correlation analysis
- ✅ Statistical significance interpretation

### Data Science
- ✅ Exploratory data analysis
- ✅ Data cleaning and preprocessing
- ✅ Feature engineering
- ✅ Class imbalance handling
- ✅ Model selection and comparison

### Machine Learning
- ✅ Logistic regression
- ✅ Ensemble methods (Random Forest, Gradient Boosting)
- ✅ Cross-validation
- ✅ Hyperparameter awareness
- ✅ Model evaluation metrics

### Communication
- ✅ Clear code documentation
- ✅ Professional visualizations
- ✅ Business-focused insights
- ✅ Actionable recommendations

---

## 📝 Future Enhancements

1. **Advanced Modeling**
   - XGBoost and LightGBM implementations
   - Neural network architectures
   - Ensemble stacking methods

2. **Time Series Analysis**
   - Survival analysis for time-to-churn
   - Temporal pattern detection
   - Cohort analysis

3. **Deployment**
   - REST API for real-time predictions
   - Model monitoring dashboard
   - A/B testing framework

4. **Business Intelligence**
   - Customer segmentation (K-means, DBSCAN)
   - Customer lifetime value (CLV) modeling
   - Retention campaign ROI analysis

---

## 📧 Contact

**Portfolio:** [Your GitHub Profile]  
**LinkedIn:** [Your LinkedIn]  
**Email:** [Your Email]

---

## 📄 License

This project is created for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- Dataset: Synthetic data generated for demonstration purposes
- Inspired by real-world telecom industry churn problems
- Built following data science best practices and CRISP-DM methodology

---

**⭐ If you found this project helpful, please star this repository!**
