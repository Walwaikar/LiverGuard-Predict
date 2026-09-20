import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load Dataset
data = pd.read_csv("dataset/indian_liver_patient.csv")

print("Original Dataset Shape:", data.shape)


# 2. Handle Missing Values
data["Albumin_and_Globulin_Ratio"] = data["Albumin_and_Globulin_Ratio"].fillna(
    data["Albumin_and_Globulin_Ratio"].median()
)


# 3. Convert Gender into Numbers
data["Gender"] = data["Gender"].map({
    "Male": 1,
    "Female": 0
})


# 4. Convert Target
data["Dataset"] = data["Dataset"].map({
    1: 1,
    2: 0
})


# 5. Select Final 6 Features
features = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens"
]

X = data[features]
y = data["Dataset"]


print("\nSelected Features:")
for feature in features:
    print("-", feature)


# 6. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))


# 7. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 8. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)


# 9. Make Predictions
y_pred = model.predict(X_test)


# 10. Evaluate Model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 11. Save Model and Scaler
joblib.dump(model, "model/liver_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel saved successfully!")
print("Scaler saved successfully!")