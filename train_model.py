import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load processed data
df = pd.read_csv("data/processed_windows.csv")


# Accelerometer features only
features = [
    "x_mean", "x_std", "x_min", "x_max",
    "y_mean", "y_std", "y_min", "y_max",
    "z_mean", "z_std", "z_min", "z_max"
]


# Participants used for testing
test_participants = [
    "BK7610",
    "BU4707",
    "CC6740"
]


# Split by participant
train_df = df[~df["pid"].isin(test_participants)]
test_df = df[df["pid"].isin(test_participants)]


X_train = train_df[features]
y_train = train_df["label"]

X_test = test_df[features]
y_test = test_df["label"]


print("Training participants:", train_df["pid"].nunique())
print("Testing participants:", test_df["pid"].nunique())

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=1234,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Predictions
predictions = model.predict(X_test)


# Evaluate
accuracy = accuracy_score(y_test, predictions)

print("\nRandom Forest Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load processed data
df = pd.read_csv("data/processed_windows.csv")

# Accelerometer features
features = [
    "x_mean", "x_std", "x_min", "x_max",
    "y_mean", "y_std", "y_min", "y_max",
    "z_mean", "z_std", "z_min", "z_max"
]

X = df[features]
y = df["label"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=1234,
    stratify=y
)


# Decision Tree
tree = DecisionTreeClassifier(
    max_depth=6,
    random_state=1234
)

tree.fit(X_train, y_train)

tree_predictions = tree.predict(X_test)

print("Decision Tree Accuracy:")
print(accuracy_score(y_test, tree_predictions))

print("\nDecision Tree Report:")
print(classification_report(y_test, tree_predictions))


# Random Forest
forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=1234,
    n_jobs=-1
)

forest.fit(X_train, y_train)

forest_predictions = forest.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, forest_predictions))

print("\nRandom Forest Report:")
print(classification_report(y_test, forest_predictions))


# Save Random Forest
joblib.dump(forest, "model.pkl")

print("\nRandom Forest saved as model.pkl")

