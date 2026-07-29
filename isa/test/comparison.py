import pandas as pd

RUNS = {
    "uniform": "uniform/csv",
    "inverse": "inverse/csv",
    "exponential": "exponential/csv",
}

TARGET_ALG = "gap"

rows = []
for name, folder in RUNS.items():
    fp = pd.read_csv(f"{folder}/footprint_performance.csv")
    svm = pd.read_csv(f"{folder}/svm_table.csv")

    fp_row = fp[fp["Row"] == TARGET_ALG].iloc[0]
    svm_row = svm[svm["Row"] == TARGET_ALG].iloc[0]

    rows.append({
        "formula": name,
        "Area_Good": fp_row["Area_Good_Normalised"],
        "Density_Good": fp_row["Density_Good_Normalised"],
        "Purity_Good": fp_row["Purity_Good"],
        "Area_Best": fp_row["Area_Best_Normalised"],
        "Purity_Best": fp_row["Purity_Best"],
        "SVM_Accuracy": svm_row.get("CV_model_accuracy", None),
        "SVM_Precision": svm_row.get("CV_model_precision", None),
        "SVM_Recall": svm_row.get("CV_model_recall", None),
    })

df_compare = pd.DataFrame(rows).sort_values(
    by=["Purity_Good", "Area_Good"], ascending=False
)

print(df_compare.to_string(index=False))