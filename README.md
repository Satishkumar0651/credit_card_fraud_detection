# Credit Card Fraud Detection

## Project Overview
This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset is highly imbalanced, and we employ techniques such as **feature selection, data balancing (SMOTE + Undersampling), and multiple classification models** to achieve the best possible fraud detection.

---

## Steps Involved

### 1. **Data Preprocessing**
- **Feature Engineering:**
  - Extracted `Hour` from `Time` column to capture transaction timing patterns.
  - Scaled `Amount` using `StandardScaler`.
- **Missing Values:**
  - Verified that no missing values exist in the dataset.
- **Class Distribution:**
  - The dataset is highly imbalanced, with fraudulent transactions making up only **0.17%** of the data.

---

### 2. **Feature Selection**
- **ANOVA F-Test** was used to select the **top 22 most important features**.
- The **top 22 features** were identified based on their F-score and used for training.
- **Final Selected Features after ANOVA:**
- V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V14, V16, V17, V18, V19, V20, V21, V27, V28, Hour
- 
#### **XGBoost Feature Importance Optimization:**
- Further refined feature selection using feature importance from XGBoost.
- Reduced features to:
- V14, V4, V10, V12, V8, V17, V1, V18, V7, V2, V28, V6, V19, Hour, V11, V27, V3
- The refined model with these features showed **better performance in ROC AUC.**

---

### 3. **Data Balancing**
- **Hybrid Approach:**
- **Undersampling**: Reduced non-fraud cases to **10%** of the original count.
- **SMOTE Oversampling**: Increased fraud cases to **50%** of non-fraud cases.
- This ensures the model learns patterns in both fraud and non-fraud transactions effectively.

---

### 4. **Model Training & Evaluation**
#### **Models Used:**
1. **Logistic Regression**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**
4. **XGBoost** *(Final Optimized Model)*

#### **Performance Metrics:**
| Model               | ROC AUC Score | Cross Validation Score | F1 Score (Fraud) |
|---------------------|--------------|------------------------|------------------|
| **Logistic Regression** | 0.9421 | 0.9863 | 0.93 |
| **Random Forest**      | 0.9735 | 0.9980 | 0.97 |
| **KNN**                | 0.9811 | 0.9966 | 0.97 |
| **XGBoost** *(Best)* | **0.9991** | **0.9991** | **0.98** |

---

### 5. **Further Feature Reduction Using XGBoost**
- Used **XGBoost’s feature importance** to further **reduce the number of features** and **improve ROC AUC score**.
- This resulted in the final **optimized feature set**:
- V14, V4, V10, V12, V8, V17, V1, V18, V7, V2, V28, V6, V19, Hour, V11, V27, V3
- The refined model showed an **improved ROC AUC score.**

---

### 6. **Deployment using FastAPI**
- **Trained XGBoost model** was saved as `xgboost_model.pkl`.
- Created a **FastAPI** endpoint to serve predictions.
- Features are extracted and transformed correctly before making predictions.
- **Prediction Output:**
- **Fraud or Non-Fraud**
- **Fraud Probability**

---

### 7. **Testing the API**
#### **Example cURL Request:**
```bash
curl -X 'POST' \
'http://127.0.0.1:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
  "Time": 406.0,
  "V1": -2.312227,
  "V2": 1.951992,
  "V3": -1.609851,
  "V4": 3.997906,
  "V5": -0.522188,
  "V6": -1.426545,
  "V7": -2.537387,
  "V8": 1.391657,
  "V9": -2.770089,
  "V10": -2.772272,
  "V11": 3.202033,
  "V12": -2.899907,
  "V13": -0.595222,
  "V14": -4.289254,
  "V15": 0.389724,
  "V16": -1.140747,
  "V17": -2.830056,
  "V18": -0.016822,
  "V19": 0.416956,
  "V20": 0.126911,
  "V21": 0.517232,
  "V22": -0.035049,
  "V23": -0.465211,
  "V24": 0.320198,
  "V25": 0.044519,
  "V26": 0.177840,
  "V27": 0.261145,
  "V28": -0.143276,
  "Amount": -0.353229
}'

### 7. **API Response**
{
  "fraud_prediction": "Fraud",
  "fraud_probability": 1.0
}
