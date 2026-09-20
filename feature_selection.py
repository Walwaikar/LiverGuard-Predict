import pandas as pd
from itertools import combinations

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# =========================================================
# 1. LOAD DATASET
# =========================================================

data = pd.read_csv("dataset/indian_liver_patient.csv")

print("Dataset Shape:", data.shape)


# =========================================================
# 2. HANDLE MISSING VALUES
# =========================================================

data["Albumin_and_Globulin_Ratio"] = data[
    "Albumin_and_Globulin_Ratio"
].fillna(
    data["Albumin_and_Globulin_Ratio"].median()
)


# =========================================================
# 3. CONVERT GENDER
# =========================================================

data["Gender"] = data["Gender"].map({
    "Male": 1,
    "Female": 0
})


# =========================================================
# 4. CONVERT TARGET
# =========================================================

data["Dataset"] = data["Dataset"].map({
    1: 1,
    2: 0
})


# =========================================================
# 5. DEFINE ALL 10 FEATURES
# =========================================================

all_features = [
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


# =========================================================
# 6. TARGET
# =========================================================

y = data["Dataset"]


# =========================================================
# 7. CROSS-VALIDATION
# =========================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================================================
# 8. METRICS
# =========================================================

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1"
}


# =========================================================
# 9. LOGISTIC REGRESSION PIPELINE
# =========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# =========================================================
# 10. TEST ALL FEATURE COMBINATIONS
# =========================================================

results = []

total_combinations = 2 ** len(all_features) - 1

print("\n==============================================")
print("EXHAUSTIVE FEATURE SELECTION")
print("==============================================")

print(
    f"Total feature combinations to test: "
    f"{total_combinations}"
)

print("Running 5-Fold Cross-Validation...\n")


combination_number = 0


for number_of_features in range(1, len(all_features) + 1):

    print(
        f"Testing combinations with "
        f"{number_of_features} feature(s)..."
    )

    for feature_combination in combinations(
        all_features,
        number_of_features
    ):

        combination_number += 1

        X = data[list(feature_combination)]

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({

            "Number of Features": number_of_features,

            "Features": ", ".join(feature_combination),

            "Accuracy": scores[
                "test_accuracy"
            ].mean() * 100,

            "Precision": scores[
                "test_precision"
            ].mean() * 100,

            "Recall": scores[
                "test_recall"
            ].mean() * 100,

            "F1 Score": scores[
                "test_f1"
            ].mean() * 100

        })


# =========================================================
# 11. CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)


# Round metrics

for column in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]:

    results_df[column] = results_df[
        column
    ].round(2)


# =========================================================
# 12. SORT BY F1 SCORE
# =========================================================

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)


# =========================================================
# 13. DISPLAY TOP 20 MODELS
# =========================================================

print("\n==============================================")
print("TOP 20 FEATURE COMBINATIONS")
print("==============================================")

print(
    results_df.head(20).to_string(index=False)
)


# =========================================================
# 14. FIND 10-FEATURE BASELINE
# =========================================================

baseline = results_df[
    results_df["Number of Features"] == 10
].iloc[0]


baseline_f1 = baseline["F1 Score"]
baseline_accuracy = baseline["Accuracy"]
baseline_precision = baseline["Precision"]
baseline_recall = baseline["Recall"]


# =========================================================
# 15. DISPLAY BASELINE
# =========================================================

print("\n==============================================")
print("10-FEATURE BASELINE")
print("==============================================")

print(
    f"Accuracy  : {baseline_accuracy:.2f}%"
)

print(
    f"Precision : {baseline_precision:.2f}%"
)

print(
    f"Recall    : {baseline_recall:.2f}%"
)

print(
    f"F1 Score  : {baseline_f1:.2f}%"
)

print(
    f"Features  : {baseline['Features']}"
)


# =========================================================
# 16. FIND BEST MODEL FOR EACH FEATURE COUNT
# =========================================================

print("\n==============================================")
print("BEST MODEL FOR EACH FEATURE COUNT")
print("==============================================")


best_by_size = []

for number_of_features in range(
    1,
    len(all_features) + 1
):

    subset = results_df[
        results_df["Number of Features"]
        == number_of_features
    ]

    best = subset.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]

    best_by_size.append(best)


best_by_size_df = pd.DataFrame(
    best_by_size
)

print(
    best_by_size_df[
        [
            "Number of Features",
            "Features",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].to_string(index=False)
)


# =========================================================
# 17. FIND FEATURE SETS WITHIN 1% OF BASELINE F1
# =========================================================

tolerance = 1.0

acceptable_models = results_df[
    results_df["F1 Score"]
    >= baseline_f1 - tolerance
]


# =========================================================
# 18. FIND SMALLEST ACCEPTABLE MODEL
# =========================================================

smallest_feature_count = (
    acceptable_models[
        "Number of Features"
    ].min()
)

smallest_models = acceptable_models[
    acceptable_models[
        "Number of Features"
    ] == smallest_feature_count
]


best_smallest = smallest_models.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]


# =========================================================
# 19. DISPLAY RECOMMENDED REDUCED MODEL
# =========================================================

print("\n==============================================")
print("RECOMMENDED REDUCED FEATURE MODEL")
print("==============================================")

print(
    f"Baseline F1 Score : "
    f"{baseline_f1:.2f}%"
)

print(
    f"Allowed F1 Loss   : "
    f"{tolerance:.2f}%"
)

print(
    f"Minimum Features  : "
    f"{smallest_feature_count}"
)

print(
    f"Accuracy           : "
    f"{best_smallest['Accuracy']:.2f}%"
)

print(
    f"Precision          : "
    f"{best_smallest['Precision']:.2f}%"
)

print(
    f"Recall             : "
    f"{best_smallest['Recall']:.2f}%"
)

print(
    f"F1 Score           : "
    f"{best_smallest['F1 Score']:.2f}%"
)

print(
    f"Features           : "
    f"{best_smallest['Features']}"
)


# =========================================================
# 20. FIND REMOVED FEATURES
# =========================================================

selected_features = (
    best_smallest["Features"]
    .split(", ")
)

removed_features = [
    feature
    for feature in all_features
    if feature not in selected_features
]


print("\n==============================================")
print("FEATURES THAT CAN BE REMOVED")
print("==============================================")

if removed_features:

    for feature in removed_features:
        print(f"- {feature}")

else:

    print(
        "No features should be removed."
    )


# =========================================================
# 21. SAVE ALL RESULTS
# =========================================================

results_df.to_csv(
    "feature_selection_results.csv",
    index=False
)


print("\n==============================================")
print("RESULTS SAVED")
print("==============================================")

print(
    "Saved as: feature_selection_results.csv"
)


# =========================================================
# 22. FINAL CONCLUSION
# =========================================================

print("\n==============================================")
print("FINAL CONCLUSION")
print("==============================================")

if smallest_feature_count < 10:

    print(
        f"You can reduce the model from "
        f"10 features to "
        f"{smallest_feature_count} features "
        f"while keeping F1 Score within "
        f"{tolerance}% of the 10-feature baseline."
    )

else:

    print(
        "The 10-feature model should be retained."
    )

print("\nSelected features:")

for feature in selected_features:
    print(f"- {feature}")

print("\nRemoved features:")

if removed_features:

    for feature in removed_features:
        print(f"- {feature}")

else:

    print("- None")