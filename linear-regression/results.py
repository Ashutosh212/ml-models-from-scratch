
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# my custom model
from models.gradient_descent import GradientDescentLR

# sklearn model for comaprison
from sklearn.linear_model import SGDRegressor

# Load dataset
# df_path = "linear-regression/dataset/Study_vs_Score_data.csv" # single predictor
df_path = "linear-regression/dataset/Admission_Predict.csv" 

df = pd.read_csv(df_path)

# X = df.drop(['Final_Marks'], axis=1).values
# y = df['Final_Marks'].values


# print(df.head())

X = df.drop(["Chance of Admit ", "Serial No."], axis=1).values

y = df["Chance of Admit "].values

# Normalize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
y = (y - np.mean(y)) / np.std(y)

assert len(X) == len(y), "Length of X and y vector is not equal"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientDescentLR()
model.fit(X_train, y_train, alpha=0.001, batch_size=16)

y_pred_custom = model.predict(X_test)

sk_model = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01)
sk_model.fit(X_train, y_train)
y_pred_sklearn = sk_model.predict(X_test)

print("Custom Model Evaluation:")
print("  MSE:", mean_squared_error(y_test, y_pred_custom.T))
print("  R² Score:", r2_score(y_test, y_pred_custom.T))

print("Sklearn SGDRegressor Evaluation:")
print("  MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("  R² Score:", r2_score(y_test, y_pred_sklearn))



