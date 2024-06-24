import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("C:/Users/abira/Downloads/churn_Modelling.csv")

# Prepare data
X = dataset.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
y = dataset['Exited']
X_encoded = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression model
log_reg_model = LogisticRegression(max_iter=2000, random_state=42)
log_reg_model.fit(X_train_scaled, y_train)
log_reg_pred = log_reg_model.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, log_reg_pred))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, log_reg_pred))

# Train and evaluate Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# Train and evaluate Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("Gradient Boosting Confusion Matrix:")
print(confusion_matrix(y_test, gb_pred))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, gb_pred))
