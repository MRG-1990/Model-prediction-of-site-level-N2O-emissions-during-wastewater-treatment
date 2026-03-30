"""
Preprocessing and exploratory analysis extracted from Scenario_1.ipynb.
"""

from __future__ import annotations

import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore")


# ============================================================
# 1. Load Data
# ============================================================

DATA_PATH = "Scenario_1_Recorrect.csv"
TARGET = "G_N2O_r5"
TIME_COLUMN = "time"
FEATURE_COLUMNS = [
    "NH4_r7",
    "NO2_r7",
    "NO3_r7",
    "DO_r5",
    "DO_r6",
    "DO_r7",
    "NH4_r5",
    "NO3_r5",
    "TSS_r7",
    "Temp_inf",
    "Flow_inf",
    "Tnload_inf",
    TARGET,
]
PLOT_COLUMNS_WITH_TIME = [TIME_COLUMN] + FEATURE_COLUMNS

df_raw = pd.read_csv(DATA_PATH)


# ============================================================
# 2. Initial Inspection
# ============================================================

print("First rows of the raw dataset:")
print(df_raw.head())
print("\nMissing values per column:")
print(df_raw.isnull().sum())
print("\nDataFrame info:")
df_raw.info()
print("\nDescriptive statistics:")
print(df_raw.describe())


# ============================================================
# 3. Time-Series Visualization of the Main Variables
# ============================================================

fig, axes = plt.subplots(nrows=len(FEATURE_COLUMNS), ncols=1, figsize=(20, 80))

for i, feature in enumerate(FEATURE_COLUMNS):
    axes[i].plot(df_raw[TIME_COLUMN], df_raw[feature], marker="x", label=feature)
    axes[i].set_title(f"Line Plot for {feature}", fontsize=24)
    axes[i].set_xlabel(TIME_COLUMN, fontsize=20)
    axes[i].set_ylabel(feature, fontsize=20)
    axes[i].tick_params(axis="both", labelsize=18)
    axes[i].legend(fontsize=18)

plt.tight_layout()
plt.show()


# ============================================================
# 4. Original Feature Distributions
# ============================================================

num_cols = len(FEATURE_COLUMNS)
fig, axes = plt.subplots(
    nrows=(num_cols // 3) + 1,
    ncols=3,
    figsize=(15, 5 * ((num_cols // 3) + 1)),
)
axes = axes.flatten()

for i, column in enumerate(FEATURE_COLUMNS):
    sns.histplot(df_raw[column], kde=True, ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ============================================================
# 5. Min-Max Scaled Density Plots for Distribution Comparison
# ============================================================

scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler.fit_transform(df_raw[FEATURE_COLUMNS]),
    columns=FEATURE_COLUMNS,
)

plt.figure(figsize=(12, 6))
for column in FEATURE_COLUMNS:
    sns.kdeplot(df_minmax[column], label=column, fill=True)

plt.xlabel("Normalized Value")
plt.ylabel("Density")
plt.title("Distribution of All Numeric Features (Min-Max Scaled)")
plt.legend()
plt.show()


# ============================================================
# 6. Correlation Analysis
# ============================================================

sorted_features = df_raw[FEATURE_COLUMNS].corr()[TARGET].abs().sort_values(ascending=True).index
sorted_features = sorted_features.drop(TARGET).append(pd.Index([TARGET]))

df_corr = df_raw[sorted_features].copy()
corr_matrix = df_corr.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Lower Triangular Correlation Matrix (Target at Bottom)")
plt.savefig("correlation_heatmap_lower_triangle.jpg", format="jpg", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 7. Feature-vs-Target Scatter Plots
# ============================================================

SCATTER_FEATURES = [
    "Flow_inf",
    "NH4_r7",
    "Temp_inf",
    "NH4_r5",
    "NO2_r7",
    "DO_r5",
    "TSS_r7",
    "Tnload_inf",
    "NO3_r5",
    "DO_r6",
    "DO_r7",
    "NO3_r7",
]

df_scatter = df_raw[SCATTER_FEATURES + [TARGET]].copy()
rows = (len(SCATTER_FEATURES) + 2) // 3
fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
axes = axes.flatten()

for i, feature in enumerate(SCATTER_FEATURES):
    axes[i].scatter(df_scatter[feature], df_scatter[TARGET], alpha=0.2, s=1)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(TARGET)
    axes[i].set_title(f"{feature} vs {TARGET}")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ============================================================
# 8. Stationarity Check with Augmented Dickey-Fuller Test
# ============================================================


def check_adfuller(ts: pd.Series, feature_name: str) -> None:
    result = adfuller(ts.dropna(), autolag="AIC")
    print(f"Feature: {feature_name}")
    print("ADF Test Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    print("-" * 50)


for feature in FEATURE_COLUMNS:
    print(f"Checking feature: {feature}")
    check_adfuller(df_raw[feature], feature)


# ============================================================
# 9. Lag Analysis with ACF and PACF
# ============================================================

series_target = df_raw[TARGET]
lag_acf = acf(series_target, nlags=100)
lag_pacf = pacf(series_target, nlags=10, method="ols")

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle="--", color="gray")
plt.axhline(y=-1.96 / np.sqrt(len(series_target)), linestyle="--", color="gray")
plt.axhline(y=1.96 / np.sqrt(len(series_target)), linestyle="--", color="gray")
plt.title("Autocorrelation Function")

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle="--", color="gray")
plt.axhline(y=-1.96 / np.sqrt(len(series_target)), linestyle="--", color="gray")
plt.axhline(y=1.96 / np.sqrt(len(series_target)), linestyle="--", color="gray")
plt.title("Partial Autocorrelation Function")

plt.tight_layout()
plt.show()


# ============================================================
# 10. Log Transformation
# ============================================================


def log_transformation(dataframe: pd.DataFrame) -> pd.DataFrame:
    return np.log1p(dataframe)

df_log = log_transformation(df_raw.copy())

print("Log-transformed dataset:")
print(df_log.head())
print("\nDescriptive statistics after log transformation:")
print(df_log.describe())


# ============================================================
# 11. Outlier Utilities
# ============================================================


def remove_outliers(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df_filtered = dataframe.copy()
    for col in columns:
        q1 = np.percentile(df_filtered[col], 25)
        q3 = np.percentile(df_filtered[col], 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
    return df_filtered


# ============================================================
# 12. Time-Based Train/Test Split with Log-Scaled Feature Visualization
# ============================================================

df_time_split_source = pd.read_csv(DATA_PATH)

df_train_time_split = df_time_split_source[
    (df_time_split_source[TIME_COLUMN] >= 209) & (df_time_split_source[TIME_COLUMN] <= 530)
].copy()
df_test_time_split = df_time_split_source[
    (df_time_split_source[TIME_COLUMN] > 530) & (df_time_split_source[TIME_COLUMN] <= 609)
].copy()

for col in FEATURE_COLUMNS:
    df_train_time_split[col] = np.log1p(df_train_time_split[col])
    df_test_time_split[col] = np.log1p(df_test_time_split[col])

colors = cm.get_cmap("tab20", len(FEATURE_COLUMNS)).colors
fig, ax = plt.subplots(figsize=(15, 7))

for i, feature in enumerate(FEATURE_COLUMNS):
    ax.plot(df_train_time_split[TIME_COLUMN], df_train_time_split[feature], color=colors[i], alpha=0.7)
    ax.plot(df_test_time_split[TIME_COLUMN], df_test_time_split[feature], color=colors[i], alpha=0.7)

split_times = [318, 390, 462, 530]
for split_time in split_times:
    ax.axvline(x=split_time, color="black", linestyle="--", linewidth=2, alpha=0.8)

ax.set_title("Log-Scaled Features Across Time with Split Boundaries")
ax.set_xlabel(TIME_COLUMN)
ax.set_ylabel("log1p(value)")
plt.tight_layout()
plt.show()
