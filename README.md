

# Surprise Housing Price Prediction

## 📌 Project Overview

This project aims to assist **Surprise Housing**, a US-based company expanding into the Australian real estate market, by building a **machine learning model** that accurately predicts house prices.
The model will help the company make informed property investment decisions and identify the key factors influencing property value.

---

## 🎯 Objectives

* Develop a **robust machine learning model** to forecast house prices.
* Identify important features that positively or negatively impact the price.
* Apply **regularization** and **hyperparameter tuning** to improve model performance.
* Use evaluation metrics such as **MSE**, **RMSE**, and **R² Score**.

---

## 📂 Dataset

The dataset contains **1460 entries** with **81 variables**.

* **Train Data**: `Housing-project-train-data.csv` (1168 rows × 81 columns)
* **Test Data**: `Hosuing-project-test-data.csv` (292 rows × 80 columns, no target column)

The dataset includes both **numerical** and **categorical** features, with some missing values that were addressed using domain knowledge and appropriate techniques.

---

## 🛠 Project Workflow

### 1. Data Understanding & Cleaning

* Checked for missing values and handled them appropriately.
* Managed categorical and numerical variables differently.

### 2. Exploratory Data Analysis (EDA)

* Created visualizations to understand distributions, correlations, and outliers.
* Identified trends and relationships with the target variable (`SalePrice`).

### 3. Feature Engineering

* Created new features from existing data.
* Applied transformations (e.g., logarithmic scaling) to normalize skewed features.

### 4. Model Building & Regularization

* Models used: **Linear Regression**, **Lasso**, **Ridge**, **Gradient Boosting**.
* Applied **regularization** to avoid overfitting.
* Used **GridSearchCV** for hyperparameter tuning.

### 5. Model Evaluation

* Metrics: **MSE**, **RMSE**, **R² Score**.
* Selected the best-performing model based on cross-validation results.

### 6. Predictions on Test Set

* Generated predictions for the test dataset.
* Output saved to `artifacts/test_predictions.csv` in the format:

  ```
  Id,SalePrice
  337,376270.18
  ...
  ```

---

## 📊 Tools & Libraries Used

* **Python 3.x**
* **Pandas**, **NumPy** – Data manipulation
* **Matplotlib**, **Seaborn** – Visualization
* **Scikit-learn** – Machine learning models & tuning
* **Joblib** – Model saving/loading

---

## 🚀 How to Run the Project

1. **Clone the repository**

   ```bash
   git clone https://github.com/Shashidharan-2005/Housing-Project.git
   cd Surprise_Housing_Project_with_EDA_and_Tuning_FIXED_Full
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**
   Open the notebook in Jupyter and execute cells:

   ```
   notebooks/EDA_and_Model_with_Fixed_EDA_and_Tuning.ipynb
   ```

4. **Generate Predictions**
   Run the training script:

   ```bash
   python src/train_fast.py
   ```

---

## 📌 Project Structure

```
Surprise_Housing_Project_with_EDA_and_Tuning_FIXED_Full/
│-- app.py
│-- requirements.txt
│-- README.md
│-- Housing-project-train-data.csv
│-- Hosuing-project-test-data.csv
│-- artifacts/
│   ├── best_model_GradientBoosting.joblib
│   ├── test_predictions.csv
│-- notebooks/
│   ├── EDA_and_Model_with_Fixed_EDA_and_Tuning.ipynb
│-- src/
│   ├── train_fast.py
│-- templates/
│   ├── index.html
```

---

