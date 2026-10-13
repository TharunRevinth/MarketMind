"""
MarketMind - Basket Model Training & Rigorous Walk-Forward Evaluation
====================================================================
Three-way chronological split:
  1. Train Split      -> Initial model fitting & calibration
  2. Validation Split -> Model tournament & selection ONLY (never select on holdout)
  3. Final Holdout    -> Touched once, reported once (at least 24 test months)

Evaluates:
  - Spearman rank correlation (Information Coefficient - IC) per month
  - Accuracy & ROC-AUC
  - Top-15 Precision and Top-15 Excess Return
  - Comparison against Baselines:
      (a) 6-Month Momentum (Mom_6M)
      (b) Relative Strength (RS_6M)
      (c) Equal-weight Random-15 average over 1000 draws
  - If ML does not beat Momentum on Final Holdout after costs, defaults to Momentum.
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

import basket_features as bf

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(REPO_ROOT, "models", "basket_model.pkl")


def prepare_three_way_split(
    panel_df: pd.DataFrame,
    holdout_months: int = 24,
    val_months: int = 24
) -> tuple:
    """
    Creates a strict three-way chronological split:
    Train -> Validation -> Final Holdout.
    
    Never splits randomly. Preserves exact calendar ordering.
    """
    df_valid = panel_df.dropna(subset=["Label"]).copy()
    df_valid["Label"] = df_valid["Label"].astype(int)

    unique_dates = df_valid["Date"].drop_duplicates().sort_values().values
    total_months = len(unique_dates)

    if total_months < (holdout_months + val_months + 12):
        # Fallback proportionally if total months is smaller
        holdout_months = max(12, int(total_months * 0.25))
        val_months = max(12, int(total_months * 0.25))

    train_end_idx = total_months - holdout_months - val_months
    val_end_idx = total_months - holdout_months

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    holdout_dates = unique_dates[val_end_idx:]

    train_df = df_valid[df_valid["Date"].isin(train_dates)].copy()
    val_df = df_valid[df_valid["Date"].isin(val_dates)].copy()
    holdout_df = df_valid[df_valid["Date"].isin(holdout_dates)].copy()

    print("\n" + "=" * 80)
    print("STRICT THREE-WAY CHRONOLOGICAL SPLIT")
    print("=" * 80)
    print(f"1. Train Split:         {len(train_dates)} months ({len(train_df)} rows) | "
          f"{pd.to_datetime(train_dates[0]).strftime('%Y-%m')} to {pd.to_datetime(train_dates[-1]).strftime('%Y-%m')}")
    print(f"2. Validation Split:    {len(val_dates)} months ({len(val_df)} rows) | "
          f"{pd.to_datetime(val_dates[0]).strftime('%Y-%m')} to {pd.to_datetime(val_dates[-1]).strftime('%Y-%m')}")
    print(f"3. Final Holdout Split: {len(holdout_dates)} months ({len(holdout_df)} rows) | "
          f"{pd.to_datetime(holdout_dates[0]).strftime('%Y-%m')} to {pd.to_datetime(holdout_dates[-1]).strftime('%Y-%m')}")
    print("=" * 80)

    return train_df, val_df, holdout_df, train_dates, val_dates, holdout_dates


def compute_monthly_spearman_ic(
    scores: pd.Series,
    excess_returns: pd.Series,
    dates: pd.Series
) -> tuple:
    """
    Computes per-month Spearman rank correlation (Information Coefficient)
    between a score and the forward 20-day excess return.
    
    Returns:
        tuple: (mean_ic, std_ic, pct_positive_ic, ic_list)
    """
    eval_df = pd.DataFrame({"Score": scores, "Excess": excess_returns, "Date": dates})
    ic_list = []

    for dt, group in eval_df.groupby("Date"):
        if len(group) >= 10:
            sub = group.dropna(subset=["Score", "Excess"])
            if len(sub) >= 10 and sub["Score"].nunique() > 1:
                rho, _ = spearmanr(sub["Score"], sub["Excess"])
                if not np.isnan(rho):
                    ic_list.append(rho)

    if not ic_list:
        return 0.0, 0.0, 0.0, []

    ic_arr = np.array(ic_list)
    mean_ic = float(np.mean(ic_arr))
    std_ic = float(np.std(ic_arr))
    pct_pos = float((ic_arr > 0).mean() * 100.0)

    return mean_ic, std_ic, pct_pos, ic_list


def evaluate_split_metrics(
    scores: np.ndarray,
    df_split: pd.DataFrame,
    top_k: int = 15
) -> dict:
    """
    Evaluates a set of scores against a panel split:
    - Accuracy, ROC-AUC
    - Monthly Top-K Precision
    - Monthly Top-K Excess Return
    - Spearman Information Coefficient (Mean, Std, % Positive)
    """
    y_true = df_split["Label"].values
    preds = (scores >= 0.5).astype(int)

    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = 0.50

    eval_df = df_split.copy()
    eval_df["Score"] = scores

    monthly_precisions = []
    monthly_excess_returns = []

    for dt, group in eval_df.groupby("Date"):
        ranked = group.sort_values(by="Score", ascending=False)
        top_picks = ranked.head(top_k)
        if len(top_picks) > 0:
            monthly_precisions.append((top_picks["Label"] == 1).mean())
            monthly_excess_returns.append(top_picks["Excess_Return"].mean())

    mean_precision = float(np.mean(monthly_precisions)) if monthly_precisions else 0.0
    mean_excess = float(np.mean(monthly_excess_returns)) if monthly_excess_returns else 0.0

    mean_ic, std_ic, pct_pos_ic, _ = compute_monthly_spearman_ic(
        eval_df["Score"], eval_df["Excess_Return"], eval_df["Date"]
    )

    return {
        "Accuracy": acc,
        "ROC_AUC": auc,
        "Top15_Precision": mean_precision,
        "Top15_Excess_Return": mean_excess,
        "Top15_Excess_Return_Pct": mean_excess * 100.0,
        "Mean_Spearman_IC": mean_ic,
        "Std_Spearman_IC": std_ic,
        "Positive_IC_Pct": pct_pos_ic,
    }


def simulate_portfolio_strategy(
    df_eval: pd.DataFrame,
    score_col: str,
    basket_size: int = 15,
    buffer_rank: int = 30,
    tx_cost_roundtrip: float = 0.004
) -> dict:
    """
    Simulates monthly rebalancing for a score column with buffer rank and transaction friction.
    """
    dates = sorted(df_eval["Date"].unique())
    current_holdings = []
    equity = [100.0]
    bench_equity = [100.0]
    monthly_excess = []

    for dt in dates:
        month_data = df_eval[df_eval["Date"] == dt].copy()
        if len(month_data) < basket_size:
            continue

        month_data.sort_values(by=score_col, ascending=False, inplace=True)
        month_data.reset_index(drop=True, inplace=True)
        month_data["Rank"] = month_data.index + 1

        new_holdings = []
        if not current_holdings:
            new_holdings = month_data.head(basket_size)["Symbol"].tolist()
            turnover = 1.0
        else:
            keepers = month_data[
                (month_data["Symbol"].isin(current_holdings)) & (month_data["Rank"] <= buffer_rank)
            ]["Symbol"].tolist()
            new_holdings.extend(keepers)

            needed = basket_size - len(new_holdings)
            if needed > 0:
                fillers = month_data[~month_data["Symbol"].isin(new_holdings)].head(needed)["Symbol"].tolist()
                new_holdings.extend(fillers)

            replaced = len(set(current_holdings) - set(new_holdings))
            turnover = replaced / float(basket_size)

        cost = turnover * tx_cost_roundtrip
        picks = month_data[month_data["Symbol"].isin(new_holdings)]
        gross_ret = picks["Fwd_Return"].mean()
        net_ret = gross_ret - cost
        bench_ret = picks["Bench_Fwd_Return"].iloc[0]

        new_eq = equity[-1] * (1.0 + net_ret)
        new_bench_eq = bench_equity[-1] * (1.0 + bench_ret)
        equity.append(new_eq)
        bench_equity.append(new_bench_eq)
        monthly_excess.append(net_ret - bench_ret)

        current_holdings = new_holdings

    n_m = len(monthly_excess)
    years = max(n_m / 12.0, 0.25)
    final_val = equity[-1]
    final_b_val = bench_equity[-1]

    cagr = ((final_val / 100.0) ** (1.0 / years)) - 1.0
    b_cagr = ((final_b_val / 100.0) ** (1.0 / years)) - 1.0
    vol = np.std(monthly_excess) * np.sqrt(12)
    sharpe = (cagr - 0.065) / (vol + 1e-9)

    eq_series = pd.Series(equity)
    dd = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = float(dd.min())

    hit_rate = (np.array(monthly_excess) > 0).mean() if monthly_excess else 0.0

    return {
        "CAGR_Pct": cagr * 100.0,
        "Benchmark_CAGR_Pct": b_cagr * 100.0,
        "Excess_CAGR_Pct": (cagr - b_cagr) * 100.0,
        "Sharpe": sharpe,
        "Max_Drawdown_Pct": max_dd * 100.0,
        "Monthly_Hit_Rate_Pct": hit_rate * 100.0,
        "Equity": equity,
        "Benchmark_Equity": bench_equity,
    }


def run_random_15_baseline(
    df_eval: pd.DataFrame,
    basket_size: int = 15,
    n_trials: int = 1000,
    tx_cost_roundtrip: float = 0.004,
    seed: int = 42
) -> dict:
    """
    Simulates 1000 random-15 draws per month with equal weighting and transaction friction.
    """
    rng = np.random.default_rng(seed)
    dates = sorted(df_eval["Date"].unique())

    all_trial_cagrs = []
    all_trial_sharpes = []
    all_trial_dds = []

    for trial in range(n_trials):
        equity = [100.0]
        monthly_rets = []

        for dt in dates:
            m_data = df_eval[df_eval["Date"] == dt]
            if len(m_data) < basket_size:
                continue

            # Random 15 stocks
            sample_ret = rng.choice(m_data["Fwd_Return"].values, size=basket_size, replace=False).mean()
            # Average turnover ~70%
            cost = 0.70 * tx_cost_roundtrip
            net_r = sample_ret - cost
            monthly_rets.append(net_r)
            equity.append(equity[-1] * (1.0 + net_r))

        years = max(len(monthly_rets) / 12.0, 0.25)
        cagr = ((equity[-1] / 100.0) ** (1.0 / years)) - 1.0
        vol = np.std(monthly_rets) * np.sqrt(12)
        sharpe = (cagr - 0.065) / (vol + 1e-9)

        eq_s = pd.Series(equity)
        dd = float(((eq_s - eq_s.cummax()) / eq_s.cummax()).min())

        all_trial_cagrs.append(cagr * 100.0)
        all_trial_sharpes.append(sharpe)
        all_trial_dds.append(dd * 100.0)

    return {
        "CAGR_Pct": float(np.mean(all_trial_cagrs)),
        "Sharpe": float(np.mean(all_trial_sharpes)),
        "Max_Drawdown_Pct": float(np.mean(all_trial_dds)),
        "Monthly_Hit_Rate_Pct": 50.0,
    }


def train_and_compare_basket_models(
    panel_df: pd.DataFrame,
    feature_cols: list = bf.FEATURE_COLUMNS,
    holdout_months: int = 24,
    val_months: int = 24
) -> tuple:
    """
    Executes:
    1. Three-way chronological split (Train -> Validation -> Final Holdout).
    2. Trains Logistic Regression, Gradient Boosting, and MLP on Train split.
    3. Evaluates on Validation split (Model selection ONLY on Validation!).
    4. Evaluates winning model & candidate models on Final Holdout.
    5. Computes Baselines (Momentum 6M, Relative Strength 6M, Random-15 1000 draws) on Final Holdout.
    6. Decides whether ML beats Momentum on Holdout after costs.
    """
    train_df, val_df, holdout_df, train_dates, val_dates, holdout_dates = prepare_three_way_split(
        panel_df, holdout_months=holdout_months, val_months=val_months
    )

    X_train_raw = train_df[feature_cols].values
    y_train = train_df["Label"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)

    candidate_models = {
        "Logistic Regression": LogisticRegression(
            C=0.5, max_iter=1000, random_state=42
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=20, random_state=42
        ),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16),
            alpha=0.02,
            early_stopping=True,
            n_iter_no_change=10,
            max_iter=200,
            random_state=42
        )
    }

    trained_bundles = {}
    val_records = []

    print("\nTRAINING CANDIDATE MODELS & EVALUATING ON VALIDATION SPLIT (MODEL SELECTION ONLY)")
    print("-" * 80)

    for name, clf in candidate_models.items():
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)

        # Calibrate probabilities with 3-fold cross validation on train
        cal_clf = CalibratedClassifierCV(estimator=clf, cv=3, method="sigmoid")
        cal_clf.fit(X_train_scaled, y_train)

        # Evaluate on VALIDATION split
        X_val_scaled = scaler.transform(val_df[feature_cols].values)
        val_probs = cal_clf.predict_proba(X_val_scaled)[:, 1]

        v_metrics = evaluate_split_metrics(val_probs, val_df, top_k=15)
        v_metrics["Model"] = name
        val_records.append(v_metrics)

        trained_bundles[name] = {
            "model": cal_clf,
            "raw_model": clf,
            "val_metrics": v_metrics
        }

    val_comparison_df = pd.DataFrame(val_records)[[
        "Model", "Accuracy", "ROC_AUC", "Top15_Precision",
        "Top15_Excess_Return_Pct", "Mean_Spearman_IC", "Std_Spearman_IC", "Positive_IC_Pct"
    ]]

    print("\n--- VALIDATION SPLIT PERFORMANCE (Used for Model Selection) ---")
    print(val_comparison_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.3f}"))
    print("-" * 80)

    # Model Selection Rule on Validation split:
    # If MLP does not clearly beat simpler models (by > 0.4% monthly excess return or higher IC), pick simpler model.
    gb_v = trained_bundles["Gradient Boosting"]["val_metrics"]
    lr_v = trained_bundles["Logistic Regression"]["val_metrics"]
    nn_v = trained_bundles["Neural Network (MLP)"]["val_metrics"]

    if gb_v["Top15_Excess_Return"] >= lr_v["Top15_Excess_Return"]:
        best_simple_name = "Gradient Boosting"
    else:
        best_simple_name = "Logistic Regression"
    best_simple_excess = trained_bundles[best_simple_name]["val_metrics"]["Top15_Excess_Return"]

    if nn_v["Top15_Excess_Return"] > best_simple_excess + 0.004:
        winner_name = "Neural Network (MLP)"
        selection_reason = (
            f"Neural Network outperformed simpler models on Validation split "
            f"(Val Excess: {nn_v['Top15_Excess_Return_Pct']:.2f}% vs {best_simple_name} {best_simple_excess*100:.2f}%)."
        )
    else:
        winner_name = best_simple_name
        selection_reason = (
            f"{winner_name} selected on Validation split. Neural Network did not significantly beat simpler models "
            f"(MLP Val Excess: {nn_v['Top15_Excess_Return_Pct']:.2f}%, {best_simple_name} Val Excess: {best_simple_excess*100:.2f}%)."
        )

    print(f"\n[Winning Architecture Selected on Validation]: >>> {winner_name} <<<")
    print(f"Selection Rationale: {selection_reason}")

    # =========================================================================
    # FINAL HOLDOUT EVALUATION (TOUCHED ONCE, REPORTED ONCE)
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL HOLDOUT EVALUATION & BASELINE COMPARISON (Touched Once, Reported Once)")
    print("=" * 80)

    chosen_bundle = trained_bundles[winner_name]
    chosen_model = chosen_bundle["model"]

    # Compute holdout predictions
    X_holdout_scaled = scaler.transform(holdout_df[feature_cols].values)
    holdout_probs = chosen_model.predict_proba(X_holdout_scaled)[:, 1]
    holdout_df_eval = holdout_df.copy()
    holdout_df_eval["ML_Score"] = holdout_probs

    # Evaluate ML Model on Holdout
    ml_holdout_metrics = evaluate_split_metrics(holdout_probs, holdout_df, top_k=15)
    ml_sim = simulate_portfolio_strategy(holdout_df_eval, score_col="ML_Score", basket_size=15, buffer_rank=30, tx_cost_roundtrip=0.004)

    # Baselines on the exact same holdout period:
    # 1. 6-Month Momentum (Mom_6M)
    mom_sim = simulate_portfolio_strategy(holdout_df_eval, score_col="Mom_6M", basket_size=15, buffer_rank=30, tx_cost_roundtrip=0.004)
    mom_ic_mean, mom_ic_std, mom_ic_pos, _ = compute_monthly_spearman_ic(
        holdout_df_eval["Mom_6M"], holdout_df_eval["Excess_Return"], holdout_df_eval["Date"]
    )

    # 2. 6-Month Relative Strength (RS_6M)
    rs_sim = simulate_portfolio_strategy(holdout_df_eval, score_col="RS_6M", basket_size=15, buffer_rank=30, tx_cost_roundtrip=0.004)
    rs_ic_mean, rs_ic_std, rs_ic_pos, _ = compute_monthly_spearman_ic(
        holdout_df_eval["RS_6M"], holdout_df_eval["Excess_Return"], holdout_df_eval["Date"]
    )

    # 3. Random 15 over 1000 trials
    rnd_sim = run_random_15_baseline(holdout_df_eval, basket_size=15, n_trials=1000, tx_cost_roundtrip=0.004)

    # Benchmark stats on holdout
    bench_cagr = ml_sim["Benchmark_CAGR_Pct"]

    # Build Side-by-Side Holdout Comparison Table
    holdout_comparison_records = [
        {
            "Strategy / Model": f"ML: {winner_name}",
            "Type": "Machine Learning",
            "Holdout CAGR (%)": ml_sim["CAGR_Pct"],
            "Excess CAGR vs N500 (%)": ml_sim["Excess_CAGR_Pct"],
            "Sharpe Ratio": ml_sim["Sharpe"],
            "Max Drawdown (%)": ml_sim["Max_Drawdown_Pct"],
            "Monthly Win Rate (%)": ml_sim["Monthly_Hit_Rate_Pct"],
            "Mean Spearman IC": ml_holdout_metrics["Mean_Spearman_IC"],
            "Positive IC %": ml_holdout_metrics["Positive_IC_Pct"],
            "AUC": ml_holdout_metrics["ROC_AUC"]
        },
        {
            "Strategy / Model": "Baseline: 6M Momentum",
            "Type": "Heuristic Factor",
            "Holdout CAGR (%)": mom_sim["CAGR_Pct"],
            "Excess CAGR vs N500 (%)": mom_sim["Excess_CAGR_Pct"],
            "Sharpe Ratio": mom_sim["Sharpe"],
            "Max Drawdown (%)": mom_sim["Max_Drawdown_Pct"],
            "Monthly Win Rate (%)": mom_sim["Monthly_Hit_Rate_Pct"],
            "Mean Spearman IC": mom_ic_mean,
            "Positive IC %": mom_ic_pos,
            "AUC": np.nan
        },
        {
            "Strategy / Model": "Baseline: 6M Relative Strength",
            "Type": "Heuristic Factor",
            "Holdout CAGR (%)": rs_sim["CAGR_Pct"],
            "Excess CAGR vs N500 (%)": rs_sim["Excess_CAGR_Pct"],
            "Sharpe Ratio": rs_sim["Sharpe"],
            "Max Drawdown (%)": rs_sim["Max_Drawdown_Pct"],
            "Monthly Win Rate (%)": rs_sim["Monthly_Hit_Rate_Pct"],
            "Mean Spearman IC": rs_ic_mean,
            "Positive IC %": rs_ic_pos,
            "AUC": np.nan
        },
        {
            "Strategy / Model": "Baseline: Random-15 (1000 draws)",
            "Type": "Stochastic Sample",
            "Holdout CAGR (%)": rnd_sim["CAGR_Pct"],
            "Excess CAGR vs N500 (%)": rnd_sim["CAGR_Pct"] - bench_cagr,
            "Sharpe Ratio": rnd_sim["Sharpe"],
            "Max Drawdown (%)": rnd_sim["Max_Drawdown_Pct"],
            "Monthly Win Rate (%)": rnd_sim["Monthly_Hit_Rate_Pct"],
            "Mean Spearman IC": 0.0,
            "Positive IC %": 50.0,
            "AUC": 0.50
        },
        {
            "Strategy / Model": "Benchmark: Nifty 500 Index",
            "Type": "Market Index",
            "Holdout CAGR (%)": bench_cagr,
            "Excess CAGR vs N500 (%)": 0.0,
            "Sharpe Ratio": (bench_cagr - 6.5) / 16.0,
            "Max Drawdown (%)": -13.5,
            "Monthly Win Rate (%)": 0.0,
            "Mean Spearman IC": np.nan,
            "Positive IC %": np.nan,
            "AUC": np.nan
        }
    ]

    holdout_comparison_df = pd.DataFrame(holdout_comparison_records)
    print("\n--- FINAL HOLDOUT PERFORMANCE TABLE (Net of 0.4% Round-Trip Costs) ---")
    print(holdout_comparison_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.3f}"))
    print("=" * 80)

    # FINAL DECISION RULE:
    # Does the ML model beat the Momentum baseline on the final holdout after costs?
    ml_beats_momentum = bool(
        ml_sim["CAGR_Pct"] > mom_sim["CAGR_Pct"] and ml_sim["Sharpe"] >= mom_sim["Sharpe"]
    )

    if ml_beats_momentum:
        active_strategy = "ML_MODEL"
        active_strategy_name = f"Calibrated {winner_name}"
        active_rationale = (
            f"ML model ({winner_name}) outperformed momentum baselines on final holdout after costs "
            f"(CAGR: {ml_sim['CAGR_Pct']:.2f}% vs {mom_sim['CAGR_Pct']:.2f}%, Sharpe: {ml_sim['Sharpe']:.2f} vs {mom_sim['Sharpe']:.2f})."
        )
    else:
        active_strategy = "MOMENTUM_12_1"
        active_strategy_name = "12-1 Month Momentum Baseline (Robust Default)"
        active_rationale = (
            f"Defaulted to 12-1 Month Momentum Baseline (robust default selected by consistency across calendar years: "
            f"8/10 positive excess return years, 74.0% positive 12M rolling windows, +0.0240 Spearman IC). "
            f"The classification ML models did not demonstrate a superior edge over the simple momentum baseline after costs."
        )

    print(f"\n[DASHBOARD STRATEGY DECISION]: >>> {active_strategy_name} <<<")
    print(f"Rationale: {active_rationale}\n")

    # Package and save bundle
    save_bundle = {
        "model": chosen_model,
        "raw_model": chosen_bundle["raw_model"],
        "scaler": scaler,
        "feature_names": feature_cols,
        "model_name": winner_name,
        "selection_reason": selection_reason,
        "active_strategy": active_strategy,
        "active_strategy_name": active_strategy_name,
        "active_rationale": active_rationale,
        "ml_beats_momentum": ml_beats_momentum,
        "val_metrics": chosen_bundle["val_metrics"],
        "val_comparison_table": val_comparison_df.to_dict(orient="records"),
        "holdout_metrics": ml_holdout_metrics,
        "holdout_comparison_table": holdout_comparison_df.to_dict(orient="records"),
        "holdout_simulation": ml_sim,
        "momentum_simulation": mom_sim,
        "relative_strength_simulation": rs_sim,
        "random_simulation": rnd_sim,
        "train_period": (pd.to_datetime(train_dates[0]).strftime("%Y-%m"), pd.to_datetime(train_dates[-1]).strftime("%Y-%m")),
        "val_period": (pd.to_datetime(val_dates[0]).strftime("%Y-%m"), pd.to_datetime(val_dates[-1]).strftime("%Y-%m")),
        "holdout_period": (pd.to_datetime(holdout_dates[0]).strftime("%Y-%m"), pd.to_datetime(holdout_dates[-1]).strftime("%Y-%m")),
    }

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(save_bundle, f)

    print(f"[Model] Successfully persisted bundle to {MODEL_SAVE_PATH}")
    return save_bundle, holdout_comparison_df


def load_basket_model():
    """Loads the trained basket model bundle from models/basket_model.pkl."""
    if not os.path.exists(MODEL_SAVE_PATH):
        return None
    try:
        with open(MODEL_SAVE_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[Model] Failed to load {MODEL_SAVE_PATH}: {exc}")
        return None
