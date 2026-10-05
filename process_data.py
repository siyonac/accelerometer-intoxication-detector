import os
import glob
import pandas as pd


ACCEL_FILE = "data/raw/all_accelerometer_data_pids_13.csv"
TAC_FOLDER = "data/clean_tac"
OUTPUT_FILE = "data/processed_windows.csv"

TAC_THRESHOLD = 0.08
WINDOW_SECONDS = 10


print("Loading TAC data...")

tac_data = {}

for file in glob.glob(os.path.join(TAC_FOLDER, "*.csv")):
    pid = os.path.basename(file).replace("_clean_TAC.csv", "")

    tac = pd.read_csv(file)
    tac.columns = ["timestamp", "TAC"]

    tac = tac.sort_values("timestamp")

    tac_data[pid] = tac


print(f"Loaded TAC data for {len(tac_data)} participants.")


print("Loading accelerometer data...")

accel = pd.read_csv(
    ACCEL_FILE,
    names=["time", "pid", "x", "y", "z"],
    header=0
)

accel = accel.dropna()

print("Accelerometer readings:", len(accel))


# Convert milliseconds to seconds
accel["timestamp"] = (accel["time"] / 1000).astype("int64")

print("Assigning TAC labels...")


labeled_chunks = []


for pid, group in accel.groupby("pid"):

    if pid not in tac_data:
        continue

    group = group.sort_values("timestamp").copy()

    tac = tac_data[pid]

    # Match each accelerometer reading to the most recent TAC reading
    group = pd.merge_asof(
        group,
        tac,
        on="timestamp",
        direction="backward"
    )

    group = group.dropna(subset=["TAC"])

    # Convert TAC into binary label
    group["label"] = (
        group["TAC"] > TAC_THRESHOLD
    ).astype(int)

    labeled_chunks.append(group)


print("Combining participants...")

labeled = pd.concat(
    labeled_chunks,
    ignore_index=True
)


print("Creating 10-second windows...")


labeled["window"] = (
    labeled["timestamp"] // WINDOW_SECONDS
).astype(int)


processed = labeled.groupby(
    ["pid", "window"]
).agg(
    x_mean=("x", "mean"),
    x_std=("x", "std"),
    x_min=("x", "min"),
    x_max=("x", "max"),

    y_mean=("y", "mean"),
    y_std=("y", "std"),
    y_min=("y", "min"),
    y_max=("y", "max"),

    z_mean=("z", "mean"),
    z_std=("z", "std"),
    z_min=("z", "min"),
    z_max=("z", "max"),

    label=("label", "mean")
).reset_index()


# Only keep windows that are clearly one class
processed = processed[
    (processed["label"] == 0) |
    (processed["label"] == 1)
].copy()

processed["label"] = processed["label"].astype(int)


processed.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nProcessing complete!")
print("Processed windows:", len(processed))

print("\nClass distribution:")
print(processed["label"].value_counts())

print("\nSaved to:", OUTPUT_FILE)