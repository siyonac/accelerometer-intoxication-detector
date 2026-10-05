import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score


# Load processed data
df = pd.read_csv("data/processed_windows.csv")


features = [
    "x_mean", "x_std", "x_min", "x_max",
    "y_mean", "y_std", "y_min", "y_max",
    "z_mean", "z_std", "z_min", "z_max"
]


participants = df["pid"].unique()

results = []


for participant in participants:

    train_df = df[df["pid"] != participant]
    test_df = df[df["pid"] == participant]

    X_train = train_df[features]
    y_train = train_df["label"]

    X_test = test_df[features]
    y_test = test_df["label"]

    # Skip participants with only one class
    if y_test.nunique() < 2:
        print(f"Skipping {participant}: only one class present")
        continue

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=1234,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, zero_division=0)

    results.append({
        "participant": participant,
        "accuracy": accuracy,
        "intoxicated_recall": recall
    })

    print(
        f"{participant}: "
        f"accuracy={accuracy:.3f}, "
        f"intoxicated recall={recall:.3f}"
    )


results_df = pd.DataFrame(results)

print("\nOverall Results")
print("-------------------------")
print("Participants evaluated:", len(results_df))
print("Average accuracy:", results_df["accuracy"].mean())
print(
    "Average intoxicated recall:",
    results_df["intoxicated_recall"].mean()
)
