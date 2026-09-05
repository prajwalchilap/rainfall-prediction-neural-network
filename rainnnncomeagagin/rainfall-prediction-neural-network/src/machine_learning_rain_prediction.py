# Rainfall Prediction Using Neural Networks
# GitHub-clean version of the coursework implementation.
#
# Set RAIN_INPUT_DIR and RAIN_OUTPUT_DIR to override the default local
# data/raw and outputs directories.
#
# Example (Windows PowerShell):
#   $env:RAIN_INPUT_DIR="C:/path/to/raw/weather/files"
#   $env:RAIN_OUTPUT_DIR="C:/path/to/project/outputs"
#
# Example (macOS/Linux):
#   export RAIN_INPUT_DIR="/path/to/raw/weather/files"
#   export RAIN_OUTPUT_DIR="/path/to/project/outputs"

# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Step 2: Set Directories
# Configure these paths for your local copy of the raw weather dataset.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
input_dir = Path(os.getenv("RAIN_INPUT_DIR", PROJECT_ROOT / "data" / "raw"))
output_dir = Path(os.getenv("RAIN_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ Step 3: Process and Save Each Variable (March 2024 - Feb 2025)
variables = ['Tair_f_inst', 'Qair_f_inst', 'Psurf_f_inst', 'LWdown_f_tavg', 'SWdown_f_tavg',
             'Wind_f_inst', 'TVeg_tavg', 'AvgSurfT_inst', 'Rainf_tavg', 'CanopInt_inst', 'SnowDepth_inst']
for var in variables:
    var_dfs = []
    for month in range(3, 13):  # March–December 2024
        file = os.path.join(str(input_dir), f"{var}_2024{month:02d}.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            df_grouped = df.groupby(['year', 'month', 'day', 'latitude', 'longitude']).agg({var: ['mean', 'max', 'min']}).reset_index()
            df_grouped.columns = ['year', 'month', 'day', 'latitude', 'longitude',
                                  f'{var}_mean', f'{var}_max', f'{var}_min']
            var_dfs.append(df_grouped)
    for month in range(1, 3):  # January–February 2025
        file = os.path.join(str(input_dir), f"{var}_2025{month:02d}.csv")
        if os.path.exists(file):
            df = pd.read_csv(file)
            df_grouped = df.groupby(['year', 'month', 'day', 'latitude', 'longitude']).agg({var: ['mean', 'max', 'min']}).reset_index()
            df_grouped.columns = ['year', 'month', 'day', 'latitude', 'longitude',
                                  f'{var}_mean', f'{var}_max', f'{var}_min']
            var_dfs.append(df_grouped)
    if var_dfs:
        var_df = pd.concat(var_dfs, ignore_index=True)
        var_df.to_csv(os.path.join(str(output_dir), f"{var}_Mar2024_Feb2025_daily.csv"), index=False)
    print(f"Processed {var}")

# ✅ Step 4: Merge All Processed Files
merged_df = None
for var in variables:
    file = os.path.join(str(output_dir), f"{var}_Mar2024_Feb2025_daily.csv")
    if os.path.exists(file):
        df = pd.read_csv(file)
        merged_df = df if merged_df is None else merged_df.merge(df, on=['year', 'month', 'day', 'latitude', 'longitude'], how='outer')
merged_df.to_csv(os.path.join(str(output_dir), "ALL_VARIABLES_Mar2024_Feb2025_daily.csv"), index=False)
print(f"Merged dataset shape: {merged_df.shape}")

# ✅ Step 5: Null Handling
df = pd.read_csv(os.path.join(str(output_dir), "ALL_VARIABLES_Mar2024_Feb2025_daily.csv"))
df.fillna(df.mean(numeric_only=True), inplace=True)
df.to_csv(os.path.join(str(output_dir), "ALL_VARIABLES_Mar2024_Feb2025_daily.csv"), index=False)
print("Filled missing values")

# ✅ Step 6: Remove Duplicates
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]
df.to_csv(os.path.join(str(output_dir), "ALL_VARIABLES_Mar2024_Feb2025_daily.csv"), index=False)
print(f"Dropped {before - after} duplicates")


# ✅ Step 7: Save Unique Lat/Lon
unique_latitudes = df['latitude'].drop_duplicates()
unique_longitudes = df['longitude'].drop_duplicates()
unique_latlon_pairs = df[['latitude', 'longitude']].drop_duplicates()
unique_latitudes.to_csv(os.path.join(str(output_dir), "unique_latitudes.csv"), index=False)
unique_longitudes.to_csv(os.path.join(str(output_dir), "unique_longitudes.csv"), index=False)
unique_latlon_pairs.to_csv(os.path.join(str(output_dir), "unique_latlon_pairs.csv"), index=False)

# ✅ Step 8: Haversine Distance to Centroid
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

centroid_lat = df["latitude"].mean()
centroid_lon = df["longitude"].mean()
df["dist_to_centroid"] = df.apply(lambda row: haversine(row["latitude"], row["longitude"], centroid_lat, centroid_lon), axis=1)
df.to_csv(os.path.join(str(output_dir), "ALL_VARIABLES_with_haversine.csv"), index=False)
print(f"Centroid: ({centroid_lat:.2f}, {centroid_lon:.2f})")

# ✅ Step 9: Create Rain Label
df["rain"] = (df["Rainf_tavg_mean"] >= 2.78e-5).astype(int)

# ✅ Step 10: Create Week-Ahead Prediction Dataset
df["date"] = pd.to_datetime(df[["year", "month", "day"]])
df = df.sort_values(["latitude", "longitude", "date"])
df["rain_next_week"] = df.groupby(["latitude", "longitude"])["rain"].shift(-7)
df = df.dropna(subset=["rain_next_week"])
df.to_csv(os.path.join(str(output_dir), "week_ahead_dataset.csv"), index=False)
print("Saved week-ahead dataset")

# ✅ Step 11: Split Train/Val/Test
df = df.sort_values(["date", "latitude", "longitude"])
train_df = df[df["date"] < "2024-12-01"]
val_df = df[(df["date"] >= "2024-12-01") & (df["date"] < "2025-01-01")]
test_df = df[df["date"] >= "2025-01-01"]
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ✅ Step 12: Feature Engineering
features = [
    "Tair_f_inst_mean", "Qair_f_inst_mean", "Psurf_f_inst_mean", "LWdown_f_tavg_mean",
    "SWdown_f_tavg_mean", "Wind_f_inst_mean", "TVeg_tavg_mean", "AvgSurfT_inst_mean", "dist_to_centroid"
]
X_train = train_df[features]
y_train = train_df["rain_next_week"]
X_val = val_df[features]
y_val = val_df["rain_next_week"]
X_test = test_df[features]
y_test = test_df["rain_next_week"]

# ✅ Step 13: Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# ✅ Step 14: Neural Network
model = tf.keras.Sequential([
    Dense(64, activation="relu", input_shape=(len(features),)),
    Dropout(0.3),
    Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor="val_loss", patience=5)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping])
model.summary()

# ✅ Step 15: Predictions & Metrics
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba >= 0.5).astype(int)
test_outputs = pd.DataFrame({
    "true_label": y_test,
    "predicted_label": y_pred.flatten(),
    "predicted_probabilities": y_pred_proba.flatten()
})
test_outputs.to_csv(PROJECT_ROOT / "results" / "test_predictions.csv", index=False)
print("Saved test predictions")

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}
print("Test Metrics:", metrics)

# ✅ Step 16: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(PROJECT_ROOT / "results" / "confusion_matrix.png")
plt.close()

# ✅ Step 17: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(PROJECT_ROOT / "results" / "roc_curve.png")
plt.close()
