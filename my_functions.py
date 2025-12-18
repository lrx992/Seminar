
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import ruptures as rpt # For change point detection
from scipy.stats import wasserstein_distance, rankdata # To calculate "Earth Mover's Distance"
from scipy.linalg import eigh, norm
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from matplotlib.patches import Patch

################################ FUNCTIONS FOR CHANGE POINT DETECTION ###############################

# change point detection
def ruptures_changepoints_Pelt(return_series, min_len=30, pen=2, model="normal"):
    x = return_series.values
    method = rpt.Pelt(model=model, min_size = min_len).fit(x) # PELT algorithm
    change_points = method.predict(pen=pen)
    starts = [0] + change_points[:-1]
    ends   = change_points
    n = len(x)
    segs = []
    for s, e in zip(starts, ends):
            segs.append((s, e))
    return segs

################################ FUNCTIONS FOR SPECTRAL CLUSTERING ###############################

def pairwise_wasserstein_v2(segments):
    m = len(segments)
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i+1, m):
            d = wasserstein_distance(segments[i], segments[j])
            D[i, j] = D[j, i] = d
    return D

def affinity_matrix_v2(D, zero_threshold=1e-8):
    m = D.shape[0]
    if m <= 1:
        return np.zeros_like(D)

    K = int(np.ceil(np.sqrt(m)))
    K = min(K, m - 1)  

    kth = np.partition(D, K, axis=1)[:, K]
    sigma = np.maximum(kth, zero_threshold)
    A = np.exp(-(D ** 2) / (sigma[:, None] * sigma[None, :]))
    np.fill_diagonal(A, 0.0)
    return A

def L_sym_matrix_v2(A):
    deg = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_m12 = 1 / np.sqrt(np.maximum(deg, 1e-12))
    Dm12 = np.diag(d_m12)
    Lsym = np.eye(A.shape[0]) - Dm12 @ A @ Dm12
    return Lsym

def choose_k_by_eigengap_v2(L_sym, k_min=1, k_max=None):
    m = L_sym.shape[0]

    evals = eigh(L_sym, eigvals_only=True)
    evals = np.clip(evals, 0.0, None)
    gaps = evals[1:] - evals[:-1]
    k_candidates = np.arange(1, m)
    lo = max(int(k_min),2)
    hi = min(int(k_max),m-1)
    mask = (k_candidates >= lo) & (k_candidates <= hi)
    
    if not np.any(mask):
        return int(min(max(k_min,1), m-1)), evals

    k = int(k_candidates[mask][np.argmax(gaps[mask])])
    return k, evals

def spectral_clustering_v2(A, k, n_init, seed = None):
    L_sym = L_sym_matrix_v2(A)
    evals, evecs = eigh(L_sym)
    U = evecs[:,:int(k)]
    row_norms = norm(U,axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1
    Y = U / row_norms
    k_means = KMeans(n_clusters=int(k), n_init=n_init,algorithm="lloyd" ,random_state=seed)
    labels = k_means.fit_predict(Y)
    return labels

def create_DataFrame(return_data, segments, labels):
    df_return = pd.DataFrame(index=return_data.index)
    df_return['return'] = return_data.values
    df_return['segment'] = np.nan
    for i, (start, end) in enumerate(segments):
        df_return.iloc[start:end, df_return.columns.get_loc('segment')] = i
    df_return['regime'] = np.nan
    for i, label in enumerate(labels):
        df_return.loc[df_return['segment'] == i, 'regime'] = label
    return df_return

###################################### BACKTEST HELPER FUNCTIONS  #######################################

#reorder dataframe by cluster variance
def reorder_regimes_by_variance(df, return_col="return", regime_col="regime", start_at=1):
    """
    Relabel regimes so that Regime 1 has the lowest variance, Regime 2 the next, etc.
    Returns (df, mapping) where mapping maps old -> new labels.
    """
    # Compute variance per regime and sort ascending
    var_by_regime = (
        df.groupby(regime_col)[return_col]
          .var()
          .sort_values()
    )

    # Build mapping: old_label -> new_label (1,2,3,...)
    mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(var_by_regime.index, start=start_at)
    }

    df = df.copy()
    df[regime_col] = df[regime_col].map(mapping)

    return df, mapping

#rank clusters by variance, i.e. create regimes
def rank_by_variance(return_series, segs, labels, k):
    seg_var = np.array([np.var(return_series.values[s:e], ddof=1) for (s,e) in segs])
    mean_var = np.array([seg_var[labels==c].mean() for c in range(k)])
    order = np.argsort(mean_var)             
    label_rank = np.zeros_like(labels)
    for r, c in enumerate(order):
        label_rank[labels==c] = r
    return label_rank, mean_var[order]        # ranked labels per segment, sorted means


#calculate performance stats for a return series 
def performance_stats(r, freq=252):
    mu = r.mean()*freq
    sigma = r.std(ddof=1)*np.sqrt(freq)
    sharpe = mu / (sigma + 1e-12)
    downside = r[r<0]
    sortino = mu / (downside.std(ddof=1)*np.sqrt(freq) + 1e-12)
    equity = (1+r).cumprod()
    dd = equity/equity.cummax() - 1
    maxdd = dd.min()
    calmar = (mu/abs(maxdd)) if maxdd < 0 else np.nan
    return {"AnnRet": mu, "AnnVol": sigma, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd, "Calmar": calmar}

def regime_label(current_window, r_train, segments_train, vol_rank_train):
    """
    Robust version of regime_label performing 1-Wasserstein matching.

    current_window : 1D np.ndarray (returns, cleaned, length match_days)
    r_train : pd.Series of training returns (cleaned, dropna already applied)
    segments_train : list of (s,e) tuples referring to r_train indices
    vol_rank_train : array of volatility-ranks per segment
    """
    # ensure current_window is 1D float array
    cur = np.asarray(current_window, dtype=float).reshape(-1)
    if len(cur) < 3:
        return None

    best_d = 1e12
    best_label = None

    for (i, (s, e)) in enumerate(segments_train):
        seg = r_train.values[s:e]

        # ensure clean segment
        if seg is None or len(seg) < 3:
            continue
        seg = np.asarray(seg, dtype=float).reshape(-1)

        # drop NaNs
        seg = seg[~np.isnan(seg)]
        if len(seg) < 3:
            continue

        # Wasserstein distance
        try:
            d = wasserstein_distance(cur, seg)
        except Exception:
            continue

        if d < best_d:
            best_d = d
            best_label = vol_rank_train[i]

    return best_label

def transition_matrix(regime_series: pd.Series, step: int = 20) -> pd.DataFrame:
    """
    Empirical transition matrix over a given step horizon.

    Parameters
    ----------
    regime_series : pd.Series
        Series of integer regimes (e.g. 0,1,2,...) indexed by date. NaNs are ignored.
    step : int
        Horizon in index steps (e.g. 20 days). We look at pairs (r_t, r_{t+step}).

    Returns
    -------
    P : pd.DataFrame
        Row-stochastic transition matrix. P.loc[i, j] = P(regime at t+step = j | regime at t = i).
        Rows with no outgoing transitions are all zeros.
    """
    s = pd.Series(regime_series).dropna().astype(int)
    if s.size <= step:
        return pd.DataFrame()

    # allow arbitrary labels (e.g. 0,1,2 or 1,2,3,...)
    states = np.sort(s.unique())
    k = len(states)
    idx_map = {state: i for i, state in enumerate(states)}

    counts = np.zeros((k, k), dtype=int)
    # pairs (r_t, r_{t+step})
    for a, b in zip(s.iloc[:-step], s.iloc[step:]):
        i = idx_map[a]
        j = idx_map[b]
        counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = counts / row_sums
        P[~np.isfinite(P)] = 0.0  # rows with no transitions -> zeros

    return pd.DataFrame(P, index=states, columns=states)


####################################### BACKTESTING #######################################

def multiasset_regime_strategy_rolling_v2(
    returns_log: pd.DataFrame,
    cluster_action_rules: dict,
    lookback_years: float = 6.0,
    match_days: int = 20,
    rebalance_every: int = 20,
    min_seg_len: int = 30,
    pen: float = 25.0,
    k_min: int = 2,
    k_max: int = 10,
    allocation: str = "equal_weight",  # or "erc"
    erc_lookback: int = 252,
    tc_bps: float = 0.0,
    freq: int = 252,
    training_mode: str = "fixed",      # "rolling", "blockwise"
    collect_block_stats: bool = False, # NEW
):

    """
    Multi-asset regime strategy with rolling lookback and per-asset cluster->action rules,
    using Wasserstein matching of the CURRENT window (last match_days) to historical
    segments in the lookback window.

    [docstring unchanged… shortened here]
    """

    # ----- SETUP -----
    assets = list(returns_log.columns)
    n_assets = len(assets)

    rets_log = returns_log.astype(float).copy()
    rets_simple = np.exp(rets_log) - 1.0

    idx = rets_log.index
    T = len(idx)

    lookback_days = int(round(lookback_years * 252))
    
    #v2 additions
    idx = rets_log.index
    T = len(idx)
    anchor_end = lookback_days + match_days   # first time we *can* have 10y + match_days
    anchor_start = max(0, anchor_end - lookback_days)
    t0 = anchor_end

    weights = np.zeros((T, n_assets))
    port_ret = np.zeros(T)
    tc = np.zeros(T)

    
    block_stats = {} if collect_block_stats else None


    # NEW: history containers
    actions_hist = np.full((T, n_assets), np.nan)    # actions at rebalance dates
    regime_hist = np.full((T, n_assets), np.nan)     # current regime (vol-rank) at rebalance
    rebalance_flag = np.zeros(T, dtype=bool)         # rebalance occurred at t?
    rebalance_change_flag = np.zeros(T, dtype=bool)  # actions changed vs previous rebalance?

    tc_per_unit = tc_bps / 1e4
    prev_w = np.full(n_assets, 1.0 / n_assets)
    prev_actions = np.ones(n_assets)                # NEW: remember last rebalance actions

    # First time we can trade: need at least the training window + match_days
    t0 = lookback_days + match_days

    for t in range(T):
        if t < t0:
            # No trading yet: keep weights at 0, portfolio_ret = 0
            weights[t, :] = prev_w
            if t > 0:
                port_ret[t] = float(prev_w @ rets_simple.iloc[t, :].fillna(0.0).values)
            continue

        # Only rebalance every rebalance_every days
        if (t - t0) % rebalance_every != 0:
            weights[t, :] = prev_w
            port_ret[t] = float(prev_w @ rets_simple.iloc[t, :].fillna(0.0).values)
            continue

        # --- REBALANCE STEP ---
        actions = np.zeros(n_assets)
        regimes = np.full(n_assets, np.nan)   # NEW: store current regime rank per asset

        for j, asset in enumerate(assets):
            asset = assets[j]
            r_series = rets_log.iloc[:, j]

            # will be used only in blockwise mode
            block_idx = None  

            # --- TRAINING WINDOW SELECTION (depends on training_mode) ---
            if training_mode == "rolling":
                # 10-year rolling window
                train_start = t - lookback_days
                train_end = t

            elif training_mode == "fixed":
                # always the initial [anchor_start : anchor_end)
                train_start = anchor_start
                train_end = anchor_end

            elif training_mode == "blockwise":
                # Block size in days; blocks start at anchor_start
                block_size = lookback_days
                # Which block is current t in? (0,1,2,...)
                current_block = max(0, (t - t0) // block_size)
                block_start = anchor_start + current_block * block_size

                # training window: up to 10y history, but never before block_start
                train_end = t
                train_start = max(block_start, t - lookback_days)
                block_idx = int(current_block)

            else:
                raise ValueError(
                    "training_mode must be 'rolling', 'fixed', or 'blockwise'"
                )

            # Slice and drop NaNs
            r_window = r_series.iloc[train_start:train_end]
            r_train = r_window.dropna()

            # Need enough data to segment & to form match window
            if len(r_train) < 2 * min_seg_len or len(r_train) < match_days:
                actions[j] = prev_actions[j]
                continue

            # Segment training window
            segments = ruptures_changepoints_Pelt(
                r_train, min_len=min_seg_len, pen=pen, model="normal"
            )

            if len(segments) < 2:
                actions[j] = prev_actions[j]
                continue

            # Segment arrays for Wasserstein
            r_list = [
                r_train.values[s:e].astype(float).reshape(-1)
                for (s, e) in segments
            ]

            # v2 Wasserstein + affinity + Laplacian
            D = pairwise_wasserstein_v2(r_list)
            A = affinity_matrix_v2(D)
            L_sym = L_sym_matrix_v2(A)

            # Choose k via eigengap
            k, evals = choose_k_by_eigengap_v2(
                L_sym,
                k_min=k_min,
                k_max=min(k_max, len(segments)),
            )

            # Spectral clustering (v2)
            labels = spectral_clustering_v2(A, k, n_init=20, seed=None)

            # Volatility ranks per segment (0..k-1)
            vol_rank, _ = rank_by_variance(r_train, segments, labels, k)

            #per-block summary (only in blockwise mode)
            if (
                collect_block_stats
                and training_mode == "blockwise"
                and block_idx is not None
            ):
                # dict for this asset
                stats_for_asset = block_stats.setdefault(asset, {})

                # only store once per (asset, block)
                if block_idx not in stats_for_asset:
                    n_segments = len(segments)
                    n_clusters = int(k)

                    uniq, counts = np.unique(labels, return_counts=True)
                    cluster_sizes = {
                        int(u): int(c) for u, c in zip(uniq, counts)
                    }

                    # inclusive end index for training window
                    train_start_date = r_series.index[train_start]
                    train_end_date = r_series.index[train_end - 1]

                    stats_for_asset[block_idx] = {
                        "block_idx": block_idx,
                        "train_start_idx": int(train_start),
                        "train_end_idx": int(train_end - 1),
                        "train_start_date": train_start_date,
                        "train_end_date": train_end_date,
                        "n_segments": n_segments,
                        "n_clusters": n_clusters,
                        "cluster_sizes": cluster_sizes,
                    }



            # --- CURRENT DISTRIBUTION (last match_days) ---
            r_current_window = r_series.iloc[t - match_days : t].dropna()
            if len(r_current_window) < match_days:
                actions[j] = prev_actions[j]
                continue

            last_n_obs = r_current_window.values

            # --- WASSERSTEIN MATCHING TO HISTORICAL SEGMENTS ---
            # Find volatility rank of the closest historical segment
            lab = regime_label(
                last_n_obs,
                r_train,
                segments,
                vol_rank,
            )
            if lab is None:
                actions[j] = prev_actions[j]
                continue

            current_rank = int(lab)
            regimes[j] = current_rank      # NEW: store regime for this asset at this rebalance

            # Map rank to action via rules
            asset_rules = cluster_action_rules.get(asset, {})
            k_rules = asset_rules.get(k, {})
            action = float(k_rules.get(current_rank, 0.0))  # default 0 if unspecified
            actions[j] = action

        # --- PORTFOLIO CONSTRUCTION FROM ACTIONS ---
        actions = np.nan_to_num(actions, nan=1.0)

        active = np.where(actions != 0.0)[0]

        if active.size == 0:
            w_new = np.zeros(n_assets)
        else:
            if allocation == "equal_weight":
                signs = np.sign(actions[active])
                base = signs  # equal absolute weight among active assets
                denom = np.sum(np.abs(base))
                if denom == 0:
                    w_new = np.zeros(n_assets)
                else:
                    w_active = base / denom
                    w_new = np.zeros(n_assets)
                    w_new[active] = w_active

            # elif allocation == "erc":
            #     # ERC on absolute exposures, then apply signs
            #     start_cov = max(0, t - erc_lookback)
            #     window_rets = rets_simple.iloc[start_cov:t, active].dropna(how="any")
            #     if window_rets.shape[0] < 2:
            #         # fallback to equal weight across active
            #         signs = np.sign(actions[active])
            #         base = signs
            #         denom = np.sum(np.abs(base))
            #         w_active = base / denom
            #     else:
            #         cov = np.cov(window_rets.values.T)
            #         w_long = equal_risk_contribution_weights(cov)
            #         signs = np.sign(actions[active])
            #         w_active = signs * w_long

            #     w_new = np.zeros(n_assets)
            #     w_new[active] = w_active
            else:
                raise ValueError("allocation must be 'equal_weight' or 'erc'")

        # Transaction costs based on turnover
        if t == 0:
            turn = np.sum(np.abs(w_new))
        else:
            turn = np.sum(np.abs(w_new - prev_w))

        tc[t] = turn * tc_per_unit

        weights[t, :] = w_new
        prev_w = w_new.copy()

        # --- NEW: store history and flags at rebalance date ---
        rebalance_flag[t] = True
        actions_hist[t, :] = actions
        regime_hist[t, :] = regimes

        # "Interesting" rebalance = actions changed vs previous rebalance
        rebalance_change_flag[t] = not np.array_equal(actions, prev_actions)
        prev_actions = actions.copy()

        # Portfolio return net of TC
        rets_t = rets_simple.iloc[t, :].fillna(0.0).values
        gross = float(w_new @ rets_t)
        port_ret[t] = gross - tc[t]

    # Build output DataFrame
    df_out = rets_log.copy()
    for j, asset in enumerate(assets):
        df_out[f"w_{asset}"] = weights[:, j]
        df_out[f"action_{asset}"] = actions_hist[:, j]   # NEW
        df_out[f"regime_{asset}"] = regime_hist[:, j]    # NEW

    # NEW: global flags
    df_out["rebalance_flag"] = rebalance_flag
    df_out["rebalance_change_flag"] = rebalance_change_flag

    df_out["portfolio_ret"] = port_ret
    df_out["equity"] = (1.0 + df_out["portfolio_ret"]).cumprod()

    # --- mark trading start (index + date) ---
    trade_start_idx = t0
    trade_start_date = df_out.index[trade_start_idx]

    df_out.attrs["trade_start_idx"] = int(trade_start_idx)
    df_out.attrs["trade_start_date"] = trade_start_date

    # optional: equity rebased to 1 at trading start
    df_out["equity_trading"] = df_out["equity"] / df_out.loc[trade_start_date, "equity"]

    # attach block_stats (if any) to the DataFrame attrs
    if block_stats is not None:
        df_out.attrs["block_stats"] = block_stats

    # --- performance stats on trading period only ---
    stats = performance_stats(
        df_out.loc[trade_start_date:, "portfolio_ret"],
        freq=freq,
    )

    return df_out, stats

############################ FUNCTIONS FOR PLOTTING #####################################

def color_mapping(df, palette="Dark2"):
    """
    Create a mapping from unique regime labels in df['regime'] to colors.
    """
    labels_unique = sorted(df['regime'].unique())
    pal = sns.color_palette(palette, n_colors=len(labels_unique))
    return {lab: pal[i] for i, lab in enumerate(labels_unique)}

def kde_plot(
    df,
    palette="Dark2",
    per_segment=True,
    figsize=(12, 6),
    title = "KDE of SPY returns by segment (colored by label)",
    show=True,
    title_input = "",
    folder2 = "normal",
    folder1 = "synthetic_iterations",
):
    if "regime" not in df.columns:
        raise ValueError("DataFrame must contain a 'regime' column")

    color_map = color_mapping(df, palette=palette)
    labels_unique = sorted(df["regime"].unique())

    fig, ax = plt.subplots(figsize=figsize)

    if per_segment and "segment" in df.columns:
        for seg in df["segment"].unique():
            seg_df = df[df["segment"] == seg]
            returns = seg_df["return"].dropna()
            if returns.empty:
                continue
            seg_label = seg_df["regime"].iloc[0]
            sns.kdeplot(
                returns,
                ax=ax,
                color=color_map[seg_label],
                lw=0.75,
                alpha=1,
                label=f"seg {seg} (lab {seg_label})",
            )
    else:
        for lab in labels_unique:
            lab_returns = df[df["regime"] == lab]["return"].dropna()
            if lab_returns.empty:
                continue
            sns.kdeplot(
                lab_returns,
                ax=ax,
                color=color_map[lab],
                lw=0.75,
                label=f"label {lab}",
            )

    legend_handles = [Patch(color=color_map[lab], label=f"Regime {lab}") for lab in labels_unique]
    ax.legend(handles=legend_handles, title="Regime", loc="best")

    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{folder1}/{folder2}/kde_norm_{title_input}.png")

    if show:
        plt.show()
    return fig, ax

def return_plot(
    df,
    figsize=(12, 6),
    title="SPY Log Returns by Cluster",
    linewidth=1.0,
    show=True,
    folder1="synthetic_iterations",
    folder2="normal",
    title_input=""
):
    # required columns
    if "regime" not in df.columns or "return" not in df.columns:
        raise ValueError("DataFrame must contain columns 'regime' and 'return'")

    # --- use same regime → color mapping as kde_plot ---
    color_map = color_mapping(df)           # <--- NEW
    regimes = sorted(df["regime"].dropna().unique())


    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_ylabel("Log Return")
    ax.set_xlabel("Date")

    # --- plot each regime using the shared color palette ---
    for r in regimes:
        y = df["return"].where(df["regime"] == r, np.nan)
        ax.plot(df.index, y, color=color_map[r], linewidth=linewidth, label=f"Regime {r}")

    # --- segment boundary lines (unchanged) ---
    if "segment" in df.columns:
        seg = df["segment"]
        cps = df.index[seg.notna() & (seg != seg.shift(1))]
        cps = [cp for cp in cps if cp != df.index[0]]
        for cp in cps:
            ax.axvline(cp, color="black", linewidth=1, linestyle="--")

    # horizontal zero
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    # --- clean legend using color patches like KDE plot ---
    legend_handles = [Patch(color=color_map[r], label=f"Regime {r}") for r in regimes]
    ax.legend(handles=legend_handles, title="Regime", loc="upper right")

    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{folder1}/{folder2}/return_plot{title_input}.png")

    if show:
        plt.show()

    return fig, ax

def summary_stats(df):
    results = []
    for regime, group in df.groupby('regime'):
        stats = group['return'].describe().to_dict()
        stats['skewness'] = group['return'].skew()
        stats['kurtosis'] = group['return'].kurtosis()
        stats['n_obs'] = int(group['return'].count())
        stats['regime'] = regime
        results.append(stats)

    res_df = pd.DataFrame(results).set_index('regime').sort_index()

    # add a Total row with stats computed over the whole dataframe
    total_stats = df['return'].describe().to_dict()
    total_stats['skewness'] = df['return'].skew()
    total_stats['kurtosis'] = df['return'].kurtosis()
    total_stats['n_obs'] = int(df['return'].count())
    res_df.loc['Total'] = pd.Series(total_stats)

    return res_df.T

def make_synthetic_norm(seed=None, delta_std=0.001, eps_rel=0.015):
    rng = np.random.default_rng(seed)
    base_sigmas = np.array([0.25, 0.5, 1, 2, 4])
    S = int(rng.integers(20, 25))  # number of segments
    parts, gt_labels, segments = [], [], []
    prev_sidx = None
    pos = 0

    for _ in range(S):
        # choose sigma index different from the previous one
        choices = np.delete(np.arange(len(base_sigmas)), prev_sidx) if prev_sidx is not None else np.arange(len(base_sigmas))
        sidx = int(rng.choice(choices))

        L = int(rng.integers(200, 301))  # 200–300
        delta = rng.normal(0.0, delta_std)
        sigma_base = base_sigmas[sidx]
        eps = rng.normal(0.0, eps_rel * sigma_base)   # relative jitter
        sigma = max(1e-6, sigma_base + eps)
        seg = rng.normal(loc=delta, scale=sigma, size=L)  # mean ≈ 0, variance shift only

        parts.append(seg)
        gt_labels.append(sidx)
        segments.append((pos, pos + L))
        pos += L
        prev_sidx = sidx

    x = np.concatenate(parts)
    return pd.Series(x, name="return"), {"labels": np.array(gt_labels), "segments": np.array(segments, dtype=int)}

def make_synthetic_laplace(seed=None, delta_std=0.001, eps_rel=0.015):
    rng = np.random.default_rng(seed)
    base_scales = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    S = int(rng.integers(15, 20))
    parts, gt_labels, segments = [], [], []
    prev_sidx = None
    pos = 0

    for _ in range(S):
        # choose scale index different from the previous one
        choices = np.delete(np.arange(len(base_scales)), prev_sidx) if prev_sidx is not None else np.arange(len(base_scales))
        sidx = int(rng.choice(choices))

        L = int(rng.integers(200, 301))  # 200–300 (same as paper)
        delta = rng.normal(0.0, delta_std)
        b_base = base_scales[sidx]
        eps = rng.normal(0.0, eps_rel * b_base)       # relative jitter
        b = max(1e-6, b_base + eps)
        seg = rng.laplace(loc=delta, scale=b, size=L)  # mean ≈ 0, scale change only

        parts.append(seg)
        gt_labels.append(sidx)
        segments.append((pos, pos + L))
        pos += L
        prev_sidx = sidx

    x = np.concatenate(parts)
    return pd.Series(x, name="return"), {"labels": np.array(gt_labels), "segments": np.array(segments, dtype=int)}

def plot_transition_heatmap(P: pd.DataFrame, title: str,
                            figsize=(4, 3), cmap="Blues", savepath: str | None = None):
    """
    Plot a heatmap of a transition matrix P.
    """
    if P.empty:
        print(f"{title}: transition matrix is empty, skipping plot.")
        return

    plt.figure(figsize=figsize)
    sns.heatmap(P, annot=True, fmt=".2f", cmap=cmap,
                cbar=False, square=True)
    plt.title(title)
    plt.xlabel("Regime at t + step")
    plt.ylabel("Regime at t")
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()
