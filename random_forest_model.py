"""
Random Forest workflow refactored from Scenario_1.ipynb.

Pipeline order:
1. Hyperparameter Optimization Using Hyperband
2. Model Training and Test Set Evaluation
3. Overfitting Assessment Using Learning Curves
4. Feature Importance Analysis
5. Residual Analysis Over Time
6. Zero-Bias Regression Plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# ============================================================
# Configuration
# ============================================================

TARGET = "G_N2O_r5"
TIME_COLUMN = "time"
RANDOM_SEED = 42
CSV_PATH = "Scenario_1_Recorrect.csv"
USE_HYPERBAND = False

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
}


# ============================================================
# Data Preparation
# ============================================================

def load_raw_data(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def prepare_logged_dataset(
    df: pd.DataFrame,
    time_column: str = TIME_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare the modeling dataset by preserving the time column and
    applying log(x + 1) transformation to the remaining columns.
    """
    time_full = df[time_column].copy()
    df_model = df.drop(time_column, axis=1)
    log_df = np.log(df_model + 1)
    df_model = pd.DataFrame(log_df, columns=df_model.columns, index=df.index)
    return df_model, time_full


def dropna_from_train_only(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    target: str = TARGET,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove rows with missing values only from the training set.
    """
    train_combined = pd.concat([x_train, y_train], axis=1).dropna()
    x_train_clean = train_combined.drop(columns=[target])
    y_train_clean = train_combined[target]
    return x_train_clean, y_train_clean


# ============================================================
# 1. Hyperparameter Optimization Using Hyperband
# ============================================================

def sample_random_params() -> dict:
    """
    Random hyperparameter sampler preserved from the original notebook.
    """
    return {
        "n_estimators": random.randint(50, 300),
        "max_depth": random.randint(3, 15),
    }


def run_hyperband_search(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict, float]:
    """
    Perform a Hyperband-style search to find a strong Random Forest configuration.
    """
    max_iter = 81
    eta = 3
    s_max = int(np.log(max_iter) / np.log(eta))
    B = (s_max + 1) * max_iter

    best_score = float("inf")
    best_params = {}

    print("\n" + "=" * 60)
    print("1. Hyperparameter Optimization Using Hyperband")
    print("=" * 60)

    for s in reversed(range(s_max + 1)):
        n = int(np.ceil(B / max_iter / (s + 1)) * eta ** s)
        r = max_iter * eta ** (-s)
        candidates = [sample_random_params() for _ in range(n)]

        for _ in range(s + 1):
            scores = []

            for config in candidates:
                model = RandomForestRegressor(
                    n_estimators=int(r),
                    max_depth=config["max_depth"],
                    random_state=RANDOM_SEED,
                )

                x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
                    x_train,
                    y_train,
                    test_size=0.33,
                    random_state=RANDOM_SEED,
                )

                model.fit(x_train_split, y_train_split)
                y_pred = model.predict(x_val_split)
                mse = mean_squared_error(y_val_split, y_pred)
                scores.append((config, mse))

            scores.sort(key=lambda x: x[1])
            candidates = [x[0] for x in scores[: max(1, int(len(scores) / eta))]]

            if scores[0][1] < best_score:
                best_score = scores[0][1]
                best_params = scores[0][0]

            r *= eta

    print("Best hyperparameters found:")
    print(best_params)
    print(f"Best validation MSE: {best_score:.4f}")

    return best_params, best_score


# ============================================================
# Plot Utilities
# ============================================================

def plot_zero_bias_regression(
    y_true: pd.Series,
    y_pred: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    output_path: str,
):
    y_true_np = y_true.to_numpy()
    y_pred_2d = np.asarray(y_pred).reshape(-1, 1)

    reg_nobias = LinearRegression(fit_intercept=False)
    reg_nobias.fit(y_pred_2d, y_true_np)
    slope_nobias = reg_nobias.coef_[0]
    y_pred_nobias = reg_nobias.predict(y_pred_2d)

    ss_res_nobias = np.sum((y_true_np - y_pred_nobias) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2_nobias = 1 - (ss_res_nobias / ss_tot)

    slope_full, intercept_full = np.polyfit(y_true, y_pred, 1)
    y_pred_full = slope_full * y_true + intercept_full
    ss_res_full = np.sum((np.asarray(y_pred) - y_pred_full) ** 2)
    ss_tot_full = np.sum((np.asarray(y_pred) - np.mean(np.asarray(y_pred))) ** 2)
    r2_full = 1 - (ss_res_full / ss_tot_full)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.1, color="blue", label="Predicted vs Actual")
    plt.plot([0, max(y_true)], [0, max(y_true)], "r--", label="Ideal Line: y = x")

    x_line = np.linspace(0, max(y_pred_2d), 500)
    y_line_nobias = slope_nobias * x_line
    plt.plot(
        x_line,
        y_line_nobias,
        color="green",
        linewidth=2,
        label=f"Zero-Bias Line: y = {slope_nobias:.4f}x | R² = {r2_nobias:.4f}",
    )

    plt.plot(
        [],
        [],
        " ",
        label=f"Regression Line (with bias): y = {slope_full:.4f}x + {intercept_full:.4f} | R² = {r2_full:.4f}",
    )

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 3. Overfitting Assessment Using Learning Curves
# ============================================================

def plot_learning_curve_random_forest(
    model,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    output_path: str,
):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=x_train,
        y=y_train,
        cv=5,
        scoring="r2",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label="Train Score", marker="o", linestyle="-")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)

    plt.plot(train_sizes, test_mean, label="Validation Score", marker="s", linestyle="-")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    plt.ylim(0.90, 1.0)
    plt.xlabel("Training Size")
    plt.ylabel("Score (R²)")
    plt.title("Learning Curve - Random Forest")
    plt.legend()
    plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 4. Feature Importance Analysis
# ============================================================

def plot_builtin_feature_importance(
    feature_names,
    importances,
    title: str,
    output_path: Optional[str] = None,
):
    importance_data = pd.DataFrame({"name": feature_names, "importance": importances})
    importance_data = importance_data.sort_values(by="importance", ascending=False)
    importance_data["percentage"] = (
        100 * importance_data["importance"] / importance_data["importance"].sum()
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_data, x="importance", y="name")

    for idx, patch in enumerate(ax.patches):
        height = patch.get_height()
        width = patch.get_width()
        perc = importance_data["percentage"].iloc[idx]
        ax.text(width, patch.get_y() + height / 2, f"{perc:.1f}%", va="center")

    plt.title(title)
    plt.xlabel("Feature Importance (%)")
    plt.ylabel("Feature Name")

    if output_path:
        plt.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# 5. Residual Analysis Over Time
# ============================================================

def plot_residuals_over_time(
    time_values: pd.Series,
    y_true: pd.Series,
    predictions: np.ndarray,
    title: str,
    y_label: str,
    output_path: str,
    ylim: tuple[float, float] | None = (-4, 4),
):
    residual = y_true - predictions

    g = sns.jointplot(
        x=time_values,
        y=residual,
        kind="scatter",
        color="blue",
        marginal_kws={"bins": 20, "fill": True},
    )

    g.ax_joint.axhline(0, color="black", linestyle="--")
    g.fig.suptitle(
        "Residual Plot Over Time with Normal Distribution Curve (KDE)",
        y=1.02,
        fontsize=16,
    )
    g.fig.subplots_adjust(top=0.9)
    g.set_axis_labels("Time", y_label, fontsize=12)
    g.fig.set_size_inches(10, 6)

    if ylim is not None:
        g.ax_joint.set_ylim(*ylim)

    plt.suptitle(title, x=0.5, y=1.02, fontsize=16)
    g.fig.savefig(output_path, format="jpg", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Main Workflow
# ============================================================

def main():
    # ------------------------------------------------------------
    # Data Preparation
    # ------------------------------------------------------------
    df = load_raw_data(CSV_PATH)
    time_full = df[TIME_COLUMN].copy()
    df_model, _ = prepare_logged_dataset(df)

    x_train, x_test, y_train, y_test = train_test_split(
        df_model.drop(TARGET, axis=1),
        df_model[TARGET],
        random_state=RANDOM_SEED,
    )

    # ------------------------------------------------------------
    # 1. Hyperparameter Optimization Using Hyperband
    # ------------------------------------------------------------
    best_params = DEFAULT_PARAMS.copy()
    if USE_HYPERBAND:
        best_params, _ = run_hyperband_search(x_train, y_train)
    else:
        print("\n" + "=" * 60)
        print("1. Hyperparameter Optimization Using Hyperband")
        print("=" * 60)
        print("Hyperband is disabled to preserve the original notebook results.")
        print("Using original parameters:")
        print(best_params)

    # ------------------------------------------------------------
    # 2. Model Training and Test Set Evaluation
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. Model Training and Test Set Evaluation")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        random_state=RANDOM_SEED,
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    last_y_val = None
    last_predictions = None
    last_val_index = None

    for fold, (train_index, val_index) in enumerate(kf.split(x_train), 1):
        x_train_fold = x_train.iloc[train_index]
        x_val_fold = x_train.iloc[val_index]
        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        model.fit(x_train_fold, y_train_fold)

        predictions = model.predict(x_val_fold)
        mse = mean_squared_error(y_val_fold, predictions)
        r2 = r2_score(y_val_fold, predictions)
        mae = mean_absolute_error(y_val_fold, predictions)

        print(f"Fold {fold} - RandomForest MSE: {mse:.2f}")
        print(f"Fold {fold} - RandomForest R²: {r2:.2f}")
        print(f"Fold {fold} - RandomForest MAE: {mae:.2f}\n")

        last_y_val = y_val_fold
        last_predictions = predictions
        last_val_index = x_val_fold.index

    final_predictions = model.predict(x_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_r2 = r2_score(y_test, final_predictions)
    final_mae = mean_absolute_error(y_test, final_predictions)

    print("Final Test Set Metrics:")
    print(f"MSE: {final_mse:.2f}")
    print(f"R² : {final_r2:.2f}")
    print(f"MAE: {final_mae:.2f}")

    # ------------------------------------------------------------
    # 3. Overfitting Assessment Using Learning Curves
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Overfitting Assessment Using Learning Curves")
    print("=" * 60)

    plot_learning_curve_random_forest(
        model=model,
        x_train=x_train,
        y_train=y_train,
        output_path="random_forest_learning_curve.jpg",
    )

    # ------------------------------------------------------------
    # 4. Feature Importance Analysis
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. Feature Importance Analysis")
    print("=" * 60)

    plot_builtin_feature_importance(
        feature_names=model.feature_names_in_,
        importances=model.feature_importances_,
        title="Feature Importance in the Random Forest Algorithm",
        output_path="random_forest_feature_importance.jpg",
    )

    # ------------------------------------------------------------
    # 5. Residual Analysis Over Time
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("5. Residual Analysis Over Time")
    print("=" * 60)

    time_val = time_full.loc[last_val_index]
    plot_residuals_over_time(
        time_values=time_val,
        y_true=last_y_val,
        predictions=last_predictions,
        title="Random Forest over Time",
        y_label="N₂O Residuals (g-N/day)",
        output_path="random_forest_residuals_jointplot.jpg",
    )

    # ------------------------------------------------------------
    # 6. Zero-Bias Regression Plot
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("6. Zero-Bias Regression Plot")
    print("=" * 60)

    plot_zero_bias_regression(
        y_true=y_test,
        y_pred=final_predictions,
        title="Zero-Bias Regression vs Ideal Line",
        x_label="Actual N₂O emissions (kg N/hr)",
        y_label="Predicted N₂O emissions (kg N/hr)",
        output_path="random_forest_zero_bias_regression.jpg",
    )


if __name__ == "__main__":
    main()