import pandas as pd
import numpy as np

from itertools import combinations

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, recall_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/indian_liver_patient.csv")

print("Dataset Shape:", data.shape)


# ============================================================
# 2. DATA PREPROCESSING
# ============================================================

# Fill missing A/G Ratio values
data["Albumin_and_Globulin_Ratio"] = data[
    "Albumin_and_Globulin_Ratio"
].fillna(
    data["Albumin_and_Globulin_Ratio"].median()
)

# Encode Gender
data["Gender"] = data["Gender"].map({
    "Male": 1,
    "Female": 0
})

# Encode target
# 1 = Liver Disease
# 2 = No Liver Disease
data["Dataset"] = data["Dataset"].map({
    1: 1,
    2: 0
})


# ============================================================
# 3. DEFINE ALL 10 FEATURES
# ============================================================

features = [
    "Age",
    "Gender",
    "Total_Bilirubin",
    "Direct_Bilirubin",
    "Alkaline_Phosphotase",
    "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
    "Total_Protiens",
    "Albumin",
    "Albumin_and_Globulin_Ratio"
]

X = data[features]
y = data["Dataset"]


# ============================================================
# 4. MODEL
# ============================================================

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),

        ("model", LogisticRegression(
            max_iter=1000
        ))
    ])


# ============================================================
# 5. SPECIFICITY
# ============================================================

# Specificity = True Negative Rate
# TN / (TN + FP)

specificity = make_scorer(
    recall_score,
    pos_label=0
)


# ============================================================
# 6. REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# ============================================================
# 7. METRICS
# ============================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "specificity": specificity
}


# ============================================================
# 8. EVALUATE FEATURE COMBINATION
# ============================================================

def evaluate_feature_set(feature_list):

    model = create_model()

    results = cross_validate(
        model,
        data[feature_list],
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    return {
        "Feature Count": len(feature_list),

        "Features": ", ".join(feature_list),

        "Accuracy Mean":
            results["test_accuracy"].mean() * 100,

        "Accuracy Std":
            results["test_accuracy"].std() * 100,

        "Precision Mean":
            results["test_precision"].mean() * 100,

        "Precision Std":
            results["test_precision"].std() * 100,

        "Recall Mean":
            results["test_recall"].mean() * 100,

        "Recall Std":
            results["test_recall"].std() * 100,

        "F1 Mean":
            results["test_f1"].mean() * 100,

        "F1 Std":
            results["test_f1"].std() * 100,

        "Specificity Mean":
            results["test_specificity"].mean() * 100,

        "Specificity Std":
            results["test_specificity"].std() * 100
    }


# ============================================================
# 9. EVALUATE 10-FEATURE BASELINE
# ============================================================

print("\n" + "=" * 80)
print("EVALUATING 10-FEATURE BASELINE")
print("=" * 80)

baseline = evaluate_feature_set(features)

print(
    f"\n10 Features | "
    f"Accuracy: {baseline['Accuracy Mean']:.2f}% | "
    f"Precision: {baseline['Precision Mean']:.2f}% | "
    f"Recall: {baseline['Recall Mean']:.2f}% | "
    f"F1: {baseline['F1 Mean']:.2f}% | "
    f"Specificity: {baseline['Specificity Mean']:.2f}%"
)


# ============================================================
# 10. EVALUATE ALL 6-FEATURE COMBINATIONS
# ============================================================

print("\n" + "=" * 80)
print("TESTING ALL 6-FEATURE COMBINATIONS")
print("=" * 80)

results = []

six_feature_combinations = list(
    combinations(features, 6)
)

print(
    f"Total 6-feature combinations: "
    f"{len(six_feature_combinations)}"
)


for i, combination in enumerate(
    six_feature_combinations,
    start=1
):

    feature_list = list(combination)

    result = evaluate_feature_set(feature_list)

    results.append(result)

    print(
        f"[{i}/{len(six_feature_combinations)}] "
        f"F1: {result['F1 Mean']:.2f}% | "
        f"Accuracy: {result['Accuracy Mean']:.2f}% | "
        f"Recall: {result['Recall Mean']:.2f}%"
    )


# ============================================================
# 11. EVALUATE ALL 7-FEATURE COMBINATIONS
# ============================================================

print("\n" + "=" * 80)
print("TESTING ALL 7-FEATURE COMBINATIONS")
print("=" * 80)

seven_feature_combinations = list(
    combinations(features, 7)
)

print(
    f"Total 7-feature combinations: "
    f"{len(seven_feature_combinations)}"
)


for i, combination in enumerate(
    seven_feature_combinations,
    start=1
):

    feature_list = list(combination)

    result = evaluate_feature_set(feature_list)

    results.append(result)

    print(
        f"[{i}/{len(seven_feature_combinations)}] "
        f"F1: {result['F1 Mean']:.2f}% | "
        f"Accuracy: {result['Accuracy Mean']:.2f}% | "
        f"Recall: {result['Recall Mean']:.2f}%"
    )


# ============================================================
# 12. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 13. SORT BY F1 SCORE
# ============================================================

results_by_f1 = results_df.sort_values(
    by="F1 Mean",
    ascending=False
)


# ============================================================
# 14. DISPLAY TOP 10 MODELS
# ============================================================

print("\n\n")
print("=" * 120)
print("TOP 10 FEATURE COMBINATIONS BY F1 SCORE")
print("=" * 120)

top_10 = results_by_f1.head(10)

for _, row in top_10.iterrows():

    print("\n--------------------------------------------")

    print(
        f"Features ({int(row['Feature Count'])}):"
    )

    print(row["Features"])

    print(
        f"Accuracy    : "
        f"{row['Accuracy Mean']:.2f}% ± "
        f"{row['Accuracy Std']:.2f}%"
    )

    print(
        f"Precision   : "
        f"{row['Precision Mean']:.2f}% ± "
        f"{row['Precision Std']:.2f}%"
    )

    print(
        f"Recall      : "
        f"{row['Recall Mean']:.2f}% ± "
        f"{row['Recall Std']:.2f}%"
    )

    print(
        f"F1 Score    : "
        f"{row['F1 Mean']:.2f}% ± "
        f"{row['F1 Std']:.2f}%"
    )

    print(
        f"Specificity : "
        f"{row['Specificity Mean']:.2f}% ± "
        f"{row['Specificity Std']:.2f}%"
    )


# ============================================================
# 15. FIND BEST 6-FEATURE MODEL
# ============================================================

six_results = results_df[
    results_df["Feature Count"] == 6
]

best_6 = six_results.sort_values(
    by="F1 Mean",
    ascending=False
).iloc[0]


print("\n\n")
print("=" * 100)
print("BEST 6-FEATURE MODEL")
print("=" * 100)

print("\nFeatures:")
print(best_6["Features"])

print(
    f"\nAccuracy    : "
    f"{best_6['Accuracy Mean']:.2f}% ± "
    f"{best_6['Accuracy Std']:.2f}%"
)

print(
    f"Precision   : "
    f"{best_6['Precision Mean']:.2f}% ± "
    f"{best_6['Precision Std']:.2f}%"
)

print(
    f"Recall      : "
    f"{best_6['Recall Mean']:.2f}% ± "
    f"{best_6['Recall Std']:.2f}%"
)

print(
    f"F1 Score    : "
    f"{best_6['F1 Mean']:.2f}% ± "
    f"{best_6['F1 Std']:.2f}%"
)

print(
    f"Specificity : "
    f"{best_6['Specificity Mean']:.2f}% ± "
    f"{best_6['Specificity Std']:.2f}%"
)


# ============================================================
# 16. FIND BEST 7-FEATURE MODEL
# ============================================================

seven_results = results_df[
    results_df["Feature Count"] == 7
]

best_7 = seven_results.sort_values(
    by="F1 Mean",
    ascending=False
).iloc[0]


print("\n\n")
print("=" * 100)
print("BEST 7-FEATURE MODEL")
print("=" * 100)

print("\nFeatures:")
print(best_7["Features"])

print(
    f"\nAccuracy    : "
    f"{best_7['Accuracy Mean']:.2f}% ± "
    f"{best_7['Accuracy Std']:.2f}%"
)

print(
    f"Precision   : "
    f"{best_7['Precision Mean']:.2f}% ± "
    f"{best_7['Precision Std']:.2f}%"
)

print(
    f"Recall      : "
    f"{best_7['Recall Mean']:.2f}% ± "
    f"{best_7['Recall Std']:.2f}%"
)

print(
    f"F1 Score    : "
    f"{best_7['F1 Mean']:.2f}% ± "
    f"{best_7['F1 Std']:.2f}%"
)

print(
    f"Specificity : "
    f"{best_7['Specificity Mean']:.2f}% ± "
    f"{best_7['Specificity Std']:.2f}%"
)


# ============================================================
# 17. FIND MODELS WITHIN 1% OF BASELINE F1
# ============================================================

baseline_f1 = baseline["F1 Mean"]

threshold = baseline_f1 - 1.0

acceptable_models = results_df[
    results_df["F1 Mean"] >= threshold
].sort_values(
    by=[
        "Feature Count",
        "F1 Mean"
    ],
    ascending=[
        True,
        False
    ]
)


print("\n\n")
print("=" * 100)
print("MODELS WITH F1 WITHIN 1% OF 10-FEATURE BASELINE")
print("=" * 100)

print(
    f"\n10-feature baseline F1: "
    f"{baseline_f1:.2f}%"
)

print(
    f"Minimum acceptable F1: "
    f"{threshold:.2f}%"
)

for _, row in acceptable_models.head(15).iterrows():

    print("\n--------------------------------------------")

    print(
        f"Feature Count: "
        f"{int(row['Feature Count'])}"
    )

    print(
        f"F1 Score: "
        f"{row['F1 Mean']:.2f}%"
    )

    print(
        f"Accuracy: "
        f"{row['Accuracy Mean']:.2f}%"
    )

    print(
        f"Recall: "
        f"{row['Recall Mean']:.2f}%"
    )

    print(
        f"Specificity: "
        f"{row['Specificity Mean']:.2f}%"
    )

    print(
        f"Features: "
        f"{row['Features']}"
    )


# ============================================================
# 18. CHECK WHETHER GENDER IS IN BEST MODELS
# ============================================================

gender_models = results_df[
    results_df["Features"].str.contains("Gender")
].sort_values(
    by="F1 Mean",
    ascending=False
)


print("\n\n")
print("=" * 100)
print("BEST MODELS THAT INCLUDE GENDER")
print("=" * 100)

for _, row in gender_models.head(10).iterrows():

    print("\n--------------------------------------------")

    print(
        f"Feature Count: "
        f"{int(row['Feature Count'])}"
    )

    print(
        f"F1 Score: "
        f"{row['F1 Mean']:.2f}%"
    )

    print(
        f"Accuracy: "
        f"{row['Accuracy Mean']:.2f}%"
    )

    print(
        f"Precision: "
        f"{row['Precision Mean']:.2f}%"
    )

    print(
        f"Recall: "
        f"{row['Recall Mean']:.2f}%"
    )

    print(
        f"Specificity: "
        f"{row['Specificity Mean']:.2f}%"
    )

    print(
        f"Features: "
        f"{row['Features']}"
    )


# ============================================================
# 19. SAVE ALL RESULTS
# ============================================================

results_df.to_csv(
    "feature_combination_results.csv",
    index=False
)

print("\n\n")
print("=" * 100)
print("RESULTS SAVED")
print("=" * 100)

print(
    "\nFile: feature_combination_results.csv"
)