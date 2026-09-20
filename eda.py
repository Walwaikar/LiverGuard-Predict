import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load Dataset
data = pd.read_csv("dataset/indian_liver_patient.csv")

print("Dataset Shape:", data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())


# 2. Handle Missing Value
data["Albumin_and_Globulin_Ratio"] = data["Albumin_and_Globulin_Ratio"].fillna(
    data["Albumin_and_Globulin_Ratio"].median()
)


# 3. Convert Target Labels
data["Disease"] = data["Dataset"].map({
    1: "Liver Disease",
    2: "No Liver Disease"
})


# --------------------------------------------------
# GRAPH 1: Disease Distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(data=data, x="Disease")

plt.title("Liver Disease Distribution")
plt.xlabel("Disease Status")
plt.ylabel("Number of Patients")

plt.show()


# --------------------------------------------------
# GRAPH 2: Age Distribution
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.histplot(data=data, x="Age", bins=20, kde=True)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.show()


# --------------------------------------------------
# GRAPH 3: Bilirubin vs Disease
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(data=data, x="Disease", y="Total_Bilirubin")

plt.title("Total Bilirubin vs Liver Disease")
plt.xlabel("Disease Status")
plt.ylabel("Total Bilirubin")

plt.show()