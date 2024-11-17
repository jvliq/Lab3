import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

url='https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
data = pd.read_csv(url)
missing_values = data.isnull().sum()
print("Brakujące wartości w kolumnach:\n", missing_values)

threshold = len(data.columns) * 0.7
data = data.dropna(thresh=threshold)

for column in data.columns:
    if data[column].isnull().sum() > 0:
        most_frequent_value = data[column].mode()[0]
        data[column].fillna(most_frequent_value, inplace=True)

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop('score', axis=1)
y = data_encoded['score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = XGBRegressor(objective='reg:squarederror')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)

cross_val_mse = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print("Cross-validated MSE:", -cross_val_mse.mean())

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
plt.xlabel('Prawdziwe wartości')
plt.ylabel('Przewidywane wartości')
plt.title('Porównanie wartości rzeczywistych i przewidywanych')
plt.show()
