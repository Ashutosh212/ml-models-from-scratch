import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# my custom model
from model import LogisticRegression as logisitic_model

# sklearn model for comaprison
from sklearn.linear_model import LogisticRegression

# Load dataset
# df_path = "linear-regression/dataset/Study_vs_Score_data.csv" # single predictor
df_path = "logistic-regression/data/Social_Network_Ads.csv" 

df = pd.read_csv(df_path)

df['Gender'] = df['Gender'] = np.where(df['Gender']=="Male", 0, 1)
print(df.head())

X = df.drop(["User ID"], axis=1).values

y = df["Purchased"].values.astype(int)

# Normalize
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

assert len(X) == len(y), "Length of X and y vector is not equal"

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = logisitic_model()
model.fit(X_train, y_train, alpha=0.001, batch_size=8)

y_pred_custom = model.predict(X_test)

sk_model = LogisticRegression(max_iter=100)
sk_model.fit(X_train, y_train)
y_pred_sklearn = sk_model.predict(X_test)

print("Custom Model Evaluation:")
print("  MSE:", mean_squared_error(y_test, y_pred_custom))
print("  R² Score:", r2_score(y_test, y_pred_custom))

print("Sklearn SGDRegressor Evaluation:")
print("  MSE:", mean_squared_error(y_test, y_pred_sklearn))
print("  R² Score:", r2_score(y_test, y_pred_sklearn))



