# Credit_Card_Approval_Predictor
automated machine learning pipeline to predict credit card application approvals. Using a dataset with a mix of numerical and categorical features

## 📌 Project Overview
This project focuses on building an automated machine learning pipeline to predict credit card application approvals. Using a dataset with a mix of numerical and categorical features, I implemented a full end-to-end workflow—from cleaning "dirty" data to optimizing a Logistic Regression model for high-stakes decision-making.

## 🛠️ Key Technical Steps
### 1. Data Preprocessing & Cleaning
* **Handling Missing Values**: Identified hidden missing values represented as '?' characters and converted them to np.nan.
* **Intelligent Imputation**: Built a custom loop to handle different data types:
    * **Numerical columns**: Imputed using the mean.
    * **Categorical columns**: Imputed using the most frequent value (mode).
* **One-Hot Encoding**: Converted text categories into numeric dummy variables using `pd.get_dummies()` with `drop_first=True` to avoid the dummy variable trap.

### 2. Feature Engineering & Scaling
* **Train-Test Split**: Divided data (67% train, 33% test) to ensure the model could be evaluated on unseen data.
* **Scaling**: Applied `StandardScaler` using the `fit_transform()` on training data and `transform()` on test data to prevent data leakage.

### 3. Model Optimization
* **Baseline Model**: Logistic Regression.
* **Hyperparameter Tuning**: Utilized `GridSearchCV` to find the optimal settings:
    * **Best Solver**: liblinear
    * **Best Regularization (C)**: 0.1
    * **Best Tolerance (tol)**: 0.01

## 📊 Results
* **Final Model Accuracy**: 83.33% on the test set.
* **Best Cross-Validation Score**: 86.15% during the grid search.

## 💻 Technologies Used
* **Language**: Python
* **Libraries**: Pandas, NumPy, Scikit-Learn
