#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap

# Example synthetic data
# Features: Age, Dosage, Drug_Type, Previous_Trials_Success
# Target: Trial_Outcome (1 for success, 0 for failure)

np.random.seed(0)
data = pd.DataFrame({
    'Age': np.random.randint(20, 80, size=100),
    'Dosage': np.random.uniform(1.0, 10.0, size=100),
    'Drug_Type': np.random.choice(['A', 'B', 'C'], size=100),
    'Previous_Trials_Success': np.random.choice([0, 1], size=100),
    'Trial_Outcome': np.random.choice([0, 1], size=100)
})

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Drug_Type'])

# Separating features and target variable
X = data.drop('Trial_Outcome', axis=1)
y = data['Trial_Outcome']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and model evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Explainability/Interpretability with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# Summary Plot
shap.summary_plot(shap_values, X_train, plot_type="bar")

# Detailed SHAP value plot for a single instance
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1][0], X_train.iloc[0])


# In[ ]:




