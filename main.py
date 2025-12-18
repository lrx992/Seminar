#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import yfinance as yf
import ruptures as rpt # For change point detection
from scipy.stats import wasserstein_distance, rankdata # To calculate "Earth Mover's Distance"
# from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score
import my_functions as func


# %% ########################## SYNTHETIC DATA ANALYSIS - NORMAL DATA ##########################

n_runs = 100
matches = 0
mismatches = 0 
under_est = 0
over_est = 0
TP = 0
FP = 0
FN = 0
FMI = 0 
FMI_list = []
results = []

# repeat for 100 random datasets
# Normal synthetic data with Pelt, normal model
for seed in range(n_runs):
    # GENERATE DATA
    synthetic_data, true_segment_labels = func.make_synthetic_norm(seed=seed, delta_std=0.001, eps_rel=0.01)
    

    # CHANGE POINT DETECTION
    test_segments = func.ruptures_changepoints_Pelt(synthetic_data, min_len=30, pen=25, model="normal")
    match = len(test_segments) == len(true_segment_labels['segments'])
    mismatch = len(test_segments) != len(true_segment_labels['segments'])
    too_few = len(test_segments) < len(true_segment_labels['segments'])
    too_many = len(test_segments) > len(true_segment_labels['segments'])
    results.append(match)
    matches += int(match)
    mismatches += int(mismatch)
    under_est += int(too_few)
    over_est += int(too_many)
    print(f"Run {seed+1}: {match}")

    # # quick visualization of the synthetic series
    plt.figure(figsize=(10,5))

    ax1 = plt.subplot(1,1,1)
    sns.lineplot(x=np.arange(len(synthetic_data)), y=synthetic_data.values, ax=ax1, color = 'black')
    ax1.set_title("a) Synthetic change point detection")
    #ax1.set_xlabel("Index")
    ax1.set_ylabel("Synthetic returns")

    # true changepoints from synthetic generator (segment end indices)
    true_segs = np.asarray(true_segment_labels['segments'], dtype=int)
    true_cps = true_segs[:, 1][:-1]  # exclude the final end index (series end)

    # estimated changepoints from ruptures (may include series end)
    est_cps = np.asarray(test_segments)
    est_cps = est_cps[est_cps < len(synthetic_data)]  # drop any boundary equal to series length

    # draw vertical lines for changepoints
    for cp in true_cps:
        ax1.axvline(cp, color='red', linestyle='--', linewidth=1)
    for cp in est_cps:
        ax1.axvline(cp, color='blue', linestyle=':', linewidth=1)

    # legend
    handles = [Patch(color='red', label='True changepoint (segment boundary)'),
            Patch(color='blue', label='Estimated changepoint')]
    ax1.legend(handles=handles, loc='upper right')

    # ax2 = plt.subplot(2,1,2)
    # sns.histplot(synthetic_data, bins=100, kde=True, stat="density", ax=ax2, color = 'black')
    # ax2.set_title("Distribution (histogram + KDE)")
    # ax2.set_xlabel("Return")
    # ax2.set_ylabel("Density")
    plt.savefig(f"synthetic_iterations/normal/syn_data{seed+1}.png")

    plt.tight_layout()

    # CLUSTERING
    synthetic_data_list = [synthetic_data.values[s:e].astype(float).reshape(-1) for (s,e) in test_segments]
    D = func.pairwise_wasserstein_v2(synthetic_data_list)
    A = func.affinity_matrix_v2(D)
    L_sym = func.L_sym_matrix_v2(A)
    k, evals = func.choose_k_by_eigengap_v2(L_sym,k_min=2,k_max=10)
    #k = k+1
    labels = func.spectral_clustering_v2(A,k,n_init=20)
    labels = labels+1
    print(f"{k} clusters")
    
    segment_labels = pd.merge(pd.DataFrame(test_segments, columns=['segment_start','segment_end']), 
                              pd.DataFrame(labels, columns=['label']), left_index=True, right_index=True)

    if match == True:
            TP = min(k,5)
            FP = max(0,k-5)
            FN = max(0,5-k)
            FMI = np.sqrt(TP/(TP+FP) * TP/(TP+FN))
            FMI_list.append(FMI)
    
    #Show eigenvalues (gaps) using seaborn

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(1, 11), y=evals[:10], marker='o', color = 'black')  
    if k < 10:  # only if the gap is within the first 10 eigenvalues we plot
        plt.plot(
            [k, k+1],
            [evals[k-1], evals[k]],
            color="red",
            linewidth=1.5,
        )  
    plt.title("b) Maximum eigengap")
    #plt.xlabel("Index")
    plt.xticks(np.arange(1, 11))
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.savefig(f'synthetic_iterations/normal/eigenvalues_norm_{seed+1}.png')

    func.kde_plot(func.create_DataFrame(synthetic_data, test_segments, labels),
                  palette="Dark2",per_segment=True,figsize=(10,5),
                  title="c) Density of synthetic returns by cluster", show=True,
                  title_input=seed+1, folder2="normal", folder1 = "synthetic_iterations" )
    func.summary_stats(func.create_DataFrame(synthetic_data, test_segments, labels))


    

#%% #### FMI Mean ####

FMI_mean = np.mean(FMI_list)
print("Normal synthetic data with Pelt, \"normal\" model results:")
print(f"Out of {n_runs} runs: {matches} matches. {mismatches} mismatches")
print(f"Under-estimations: {under_est}, Over-estimations: {over_est}")
print(f"FMI score: {FMI_mean}")
print(f"Out of {matches} matches: {TP} true positives")

#%% ########################## SYNTHETIC DATA ANALYSIS - LAPLACIAN DATA ##########################

n_runs = 100
matches = 0
mismatches = 0 
under_est = 0
over_est = 0
TP = 0
FP = 0
FN = 0
FMI = 0 
FMI_list = []
results = []

# repeat for 100 random datasets
# Laplace synthetic data with Pelt, normal model
for seed in range(n_runs):
    # GENERATE DATA
    synthetic_data, true_segment_labels = func.make_synthetic_laplace(seed=seed, delta_std=0.001, eps_rel=0.01)
    

    # CHANGE POINT DETECTION
    test_segments = func.ruptures_changepoints_Pelt(synthetic_data, min_len=30, pen=30, model="normal")
    match = len(test_segments) == len(true_segment_labels['segments'])
    mismatch = len(test_segments) != len(true_segment_labels['segments'])
    too_few = len(test_segments) < len(true_segment_labels['segments'])
    too_many = len(test_segments) > len(true_segment_labels['segments'])
    results.append(match)
    matches += int(match)
    mismatches += int(mismatch)
    under_est += int(too_few)
    over_est += int(too_many)
    print(f"Run {seed+1}: {match}")

    # # quick visualization of the synthetic series
    plt.figure(figsize=(10,5))

    ax1 = plt.subplot(1,1,1)
    sns.lineplot(x=np.arange(len(synthetic_data)), y=synthetic_data.values, ax=ax1, color = 'black')
    ax1.set_title("a) Synthetic change point detection")
    #ax1.set_xlabel("Index")
    ax1.set_ylabel("a) Synthetic returns, Laplace distribution")

    # true changepoints from synthetic generator (segment end indices)
    true_segs = np.asarray(true_segment_labels['segments'], dtype=int)
    true_cps = true_segs[:, 1][:-1]  # exclude the final end index (series end)

    # estimated changepoints from ruptures (may include series end)
    est_cps = np.asarray(test_segments)
    est_cps = est_cps[est_cps < len(synthetic_data)]  # drop any boundary equal to series length

    # draw vertical lines for changepoints
    for cp in true_cps:
        ax1.axvline(cp, color='red', linestyle='--', linewidth=1)
    for cp in est_cps:
        ax1.axvline(cp, color='blue', linestyle=':', linewidth=1)

    # legend
    handles = [Patch(color='red', label='True changepoint (segment boundary)'),
            Patch(color='blue', label='Estimated changepoint')]
    ax1.legend(handles=handles, loc='upper right')

    # ax2 = plt.subplot(2,1,2)
    # sns.histplot(synthetic_data, bins=100, kde=True, stat="density", ax=ax2, color='black')
    # ax2.set_title("Distribution (histogram + KDE)")
    # ax2.set_xlabel("Return")
    # ax2.set_ylabel("Density")
    plt.savefig(f"synthetic_iterations/laplace/syn_data{seed+1}.png")

    plt.tight_layout()

    # CLUSTERING
    synthetic_data_list = [synthetic_data.values[s:e].astype(float).reshape(-1) for (s,e) in test_segments]
    D = func.pairwise_wasserstein_v2(synthetic_data_list)
    A = func.affinity_matrix_v2(D)
    L_sym = func.L_sym_matrix_v2(A)
    k, evals = func.choose_k_by_eigengap_v2(L_sym,k_min=2,k_max=10)
    #k = k+1
    labels = func.spectral_clustering_v2(A,k,n_init=20)
    labels = labels+1
    print(f"{k} clusters")
    
    segment_labels = pd.merge(pd.DataFrame(test_segments, columns=['segment_start','segment_end']), 
                              pd.DataFrame(labels, columns=['label']), left_index=True, right_index=True)

    if match == True:
            TP = min(k,5)
            FP = max(0,k-5)
            FN = max(0,5-k)
            FMI = np.sqrt(TP/(TP+FP) * TP/(TP+FN))
            FMI_list.append(FMI)
    
    #Show eigenvalues (gaps) using seaborn
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=np.arange(1, 11), y=evals[:10], marker='o', color = 'black')
    if k < 10:  # only if the gap is within the first 10 eigenvalues we plot
        plt.plot(
            [k, k+1],
            [evals[k-1], evals[k]],
            color="red",
            linewidth=1.5,)
    plt.title("b) Maximum eigengap, Laplace distribution")
    #plt.xlabel("Index")
    plt.xticks(np.arange(1, 11))
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.savefig(f'synthetic_iterations/laplace/eigenvalues_norm_{seed+1}.png')

    func.kde_plot(func.create_DataFrame(synthetic_data, test_segments, labels),
                  palette="Dark2",per_segment=True,figsize=(10,5),
                  title="c) Density of synthetic Laplace returns by segment", show=True,
                  title_input=seed+1, folder2="laplace", folder1 = "synthetic_iterations" )
    func.summary_stats(func.create_DataFrame(synthetic_data, test_segments, labels))

#%% #### FMI Mean ####

FMI_mean = np.mean(FMI_list)
print("Laplace synthetic data with Pelt, \"normal\" model results:")
print(f"Out of {n_runs} runs: {matches} matches. {mismatches} mismatches")
print(f"Under-estimations: {under_est}, Over-estimations: {over_est}")
print(f"FMI score: {FMI_mean}")
#print(f"Out of {matches} matches: {TP} true positives")

#%% ########################## REAL DATA ANALYSIS ##########################


tickers_LI_et_al = [
    #"^IXIC",    # NASDAQ Composite index
    "SPY",      # S&P 500 ETF
    #"BTC-USD",  # Bitcoin
    "GLD",      # Gold ETF
    "TLT",      # Long-term Treasury ETF
    #"VNQ",       # Real estate ETF (REIT)
    #"EEM",      # Emerging Markets ETF
]

data = yf.download(tickers_LI_et_al, start="2000-01-01", end="2025-12-01")["Close"]
returns = np.log(data).diff()

table_dict = {}
table_2_dict = {}
dict_trans = {}

for i in range(len(returns.columns)):
    t = returns.columns[i]
    r = returns.iloc[:, i].dropna()
    segments = func.ruptures_changepoints_Pelt(r, min_len=30, pen=22, model="normal")
    r_list = [r.values[s:e].astype(float).reshape(-1) for (s, e) in segments]
        
    #calculate the Distance matrix, D, using the Wasserstein distance
    D = func.pairwise_wasserstein_v2(r_list)
    #calculate the affinity matrix, A, the L_sym matrix and perform spectral clustering
    A = func.affinity_matrix_v2(D)
    L_sym = func.L_sym_matrix_v2(A)
    #label the clusters (and return number of unique regmies, k)
    k, evals = func.choose_k_by_eigengap_v2(L_sym, k_min=2, k_max=15)
    
    #spectral clustering
    labels = func.spectral_clustering_v2(A,k,n_init=20, seed = None)
    labels = labels+1  

    plot_evals = np.asarray(evals).ravel()  # ensure 1D

    plot_df = pd.DataFrame({
    "idx": np.arange(1, len(plot_evals) + 1),
    "eig": plot_evals
    })

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=plot_df.iloc[:10], x="idx", y="eig", marker="o", color = 'black')
    plt.plot(
    [k, k+1],
    [plot_df.loc[k-1, "eig"], plot_df.loc[k, "eig"]],
    color="red",
    linewidth=1.5,)
    plt.title(f"b) Maximum eigengap, {t}")
    #plt.xlabel("Index")
    plt.xticks(np.arange(1, 11))
    plt.ylabel("Eigenvalue")
    plt.savefig(f"real_data/eigen/eigen_fig_{t}")
    plt.show()

    # create and append summary info for current ticker
    period = f"{r.index[0].date()}_{r.index[-1].date()}"
    n_segments = len(segments)
    n_clusters = int(k)
    unique, counts = np.unique(labels, return_counts=True)
    cluster_size = dict(zip(unique.astype(int).tolist(), counts.tolist()))


    # FOR TABLE 5
    # store nested dict per ticker and period (period_1, period_2, ...)
    if t not in table_dict:
        table_dict[t] = {}


    asset_df= func.create_DataFrame(r, segments, labels)
    asset_df, regime_map = func.reorder_regimes_by_variance(asset_df)

    # FOR TRANSITION MATRIX HEATMAP
    df_trans = asset_df.copy()

    if t not in dict_trans:
        dict_trans[t] = {}

    P = func.transition_matrix(df_trans["regime"])
    dict_trans[t] = P

    # define the time blocks you care about
    period_specs = [
        ("full", asset_df.index.min(), asset_df.index.max()),
        ("2005-2009", pd.Timestamp("2005-01-01"), pd.Timestamp("2008-12-31")),
        ("2005-2010", pd.Timestamp("2009-01-01"), pd.Timestamp("2012-12-31")),
        ("2010-2015", pd.Timestamp("2013-01-01"), pd.Timestamp("2016-12-31")),
        ("2015-2020", pd.Timestamp("2017-01-01"), pd.Timestamp("2020-12-31")),
        ("2020-2025", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-01")),
    ]

    for label, start, end in period_specs:
        if label == "full":
            df_p = asset_df
        else:
            # adjust column/index if your dates are stored differently
            df_p = asset_df.loc[(asset_df.index >= start) & (asset_df.index <= end)]

        if df_p.empty:
            continue  # skip periods with no data

        # assume each row has 'segment_id' and 'cluster', one cluster per segment
        seg_cluster = df_p[["segment", "regime"]].drop_duplicates()

        n_segments_p = int(seg_cluster["segment"].nunique())
        n_clusters_p = int(seg_cluster["regime"].nunique())
        cluster_sizes_p = (
            seg_cluster.groupby("regime")["segment"]
            .nunique()
            .to_dict()
        )

        # make sure keys are plain ints
        cluster_sizes_p = {int(k): int(v) for k, v in cluster_sizes_p.items()}

        table_dict[t][label] = {
            "n_segments": n_segments_p,
            "n_clusters": n_clusters_p,
            "cluster_sizes": cluster_sizes_p,
        }
    
    if t not in table_dict:
        table_2_dict[t] = {}

    table_2_dict[t] = func.summary_stats(asset_df)

    func.return_plot(asset_df, figsize=(10,5),title=f"a) Returns by cluster, {t}",linewidth=1 ,show=True, folder1 = "real_data", folder2="returns", title_input=t) 
    func.kde_plot(asset_df,palette="Dark2",per_segment=True,figsize=(10,5),title=f"c) Segment densities of {t} returns by regime", show=True, folder1 = "real_data", folder2="kde", title_input=t)

#%% #### Transition matrix heatmaps ####


fig, axes = plt.subplots(1, len(tickers_LI_et_al), figsize=(4*len(P), 6), sharey=False)

if len(tickers_LI_et_al) == 1:
    axes = [axes]

for ax, t in zip(axes, tickers_LI_et_al):
    P = dict_trans[t]          # your transition matrix for this asset
    sns.heatmap(
        P,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,        # probabilities
        cbar=False,
        square=True,
        ax=ax,
    )
    ax.set_title(t)
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")

plt.tight_layout()
plt.savefig("real_data/transition_matrix/transition_matrices_heatmap.png")
plt.show()


#Latex Tables
for ticker in table_2_dict:
    print(f"Summary statistics for {ticker}:")
    print(table_2_dict[ticker])
    print("\n")

    #export to latex
    stats_df = pd.DataFrame(table_2_dict[ticker])
    latex_table = stats_df.to_latex(float_format="%.4f", caption=f"Summary Statistics for {ticker}", label=f"tab:summary_{ticker}")
    with open(f"real_data/tables/summary_stats_{ticker}.tex", "w") as f:
        f.write(latex_table)


# === Turn table_dict into DataFrames + LaTeX ===
for ticker, periods in table_dict.items():
    rows = []
    for period_label, info in periods.items():
        rows.append({
            "period": period_label,
            "n_segments": info["n_segments"],
            "n_clusters": info["n_clusters"],
            "cluster_sizes": info["cluster_sizes"],   # KEEP AS DICT
        })

    df_periods = pd.DataFrame(rows).set_index("period").sort_index()

    print(f"\nBlock/period summary for {ticker}:")
    print(df_periods)

    # export to LaTeX (dict printed as string)
    latex_table = df_periods.to_latex(
        escape=False,                 # allow dict printing
        caption=f"Segments and clusters by period for {ticker}",
        label=f"tab:segments_{ticker}",
    )
    with open(f"real_data/tables/segments_clusters_{ticker}.tex", "w") as f:
        f.write(latex_table)

#%% ########################## TRADING STRATEGY ##########################


#%% ####################### SPY, GLD TLT: #################

cluster_action_rules = {
    "SPY": {
        5: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
        4: {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        3: {0: 1.0, 1: 0.0, 2: 0.0}, # 1: 0.0 can be changed to 1.0 to increase returns, but decrease sharpe
        2: {0: 1.0, 1: 1.0},
        1: {0: 1.0},
    }, 
    "GLD": {
        5: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        4: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        3: {0: 0.0, 1: 1.0, 2: 1.0},
        2: {0: 1.0, 1: 1.0}, #0: 1.0 can be changed to 0.0 to increase returns, but decrease sharpe
        1: {0: 1.0},
    },   
    "TLT": {
        5: {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0},
        4: {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0},
        3: {0: 0.0, 1: 0.0, 2: 1.0},
        2: {0: 0.0, 1: 1.0},
        1: {0: 1.0},
    },

}

assets = ["SPY", "GLD", "TLT"]
returns = yf.download(assets, start="2005-01-01", end="2025-12-01")["Close"]
returns_log_SPY_GLD_TLT = np.log(returns).diff()
returns_log_sel_SPY_GLD_TLT = returns_log_SPY_GLD_TLT[assets].dropna()

df_SPY_GLD_TLT, stats_SPY_GLD_TLT = func.multiasset_regime_strategy_rolling_v2(
    returns_log=returns_log_sel_SPY_GLD_TLT,
    cluster_action_rules=cluster_action_rules,
    lookback_years=4.0,
    match_days=20,
    rebalance_every=20,
    min_seg_len=30,
    pen=15,
    k_min=2,
    k_max=5,
    allocation="equal_weight",
    erc_lookback=252,
    tc_bps=0.0,
    freq=252,
    training_mode="blockwise",
    collect_block_stats=True,
)

for df in [df_SPY_GLD_TLT]:
    assets = list(returns_log_sel_SPY_GLD_TLT.columns)
    simple_rets = np.exp(returns_log_sel_SPY_GLD_TLT) - 1.0
    w_eq = np.ones(len(assets)) / len(assets)

    eq_bh_returns = simple_rets.dot(w_eq)
    eq_bh_equity = (1 + eq_bh_returns).cumprod()
    SPY_returns = simple_rets["SPY"]
    SPY_equity  = (1 + SPY_returns).cumprod()
    GLD_returns = simple_rets["GLD"]
    GLD_equity  = (1 + GLD_returns).cumprod()
    TLT_returns = simple_rets["TLT"]
    TLT_equity  = (1 + TLT_returns).cumprod()

    start_date = df.attrs.get("trade_start_date", df.index[0])

    ret_eq  = eq_bh_returns.loc[start_date:]
    ret_SPY = SPY_returns.loc[start_date:]
    ret_GLD = GLD_returns.loc[start_date:]
    ret_TLT = TLT_returns.loc[start_date:]

    stats_SPY_GLD_TLT["eq_weight_bh"] = func.performance_stats(ret_eq,  freq=252)
    stats_SPY_GLD_TLT["SPY_bh"]       = func.performance_stats(ret_SPY, freq=252)
    stats_SPY_GLD_TLT["GLD_bh"]       = func.performance_stats(ret_GLD, freq=252)
    stats_SPY_GLD_TLT["TLT_bh"]       = func.performance_stats(ret_TLT, freq=252)

    eq_bh_equity = (1 + eq_bh_returns).cumprod()
    SPY_equity   = (1 + SPY_returns).cumprod()
    GLD_equity   = (1 + GLD_returns).cumprod()
    TLT_equity   = (1 + TLT_returns).cumprod()

    eq_bh_equity_aligned = eq_bh_equity / eq_bh_equity.loc[start_date]
    SPY_equity_aligned   = SPY_equity   / SPY_equity.loc[start_date]
    GLD_equity_aligned   = GLD_equity   / GLD_equity.loc[start_date]
    TLT_equity_aligned   = TLT_equity   / TLT_equity.loc[start_date]

    df["equity_eq_weight"] = eq_bh_equity_aligned
    df["equity_SPY"] = SPY_equity_aligned
    df["equity_GLD"] = GLD_equity_aligned
    df["equity_TLT"] = TLT_equity_aligned

    # equity plot (rebased)
    plt.figure(figsize=(11, 6))
    equity_strat = df["equity"].loc[start_date:] / df.loc[start_date, "equity"]
    equity_bh    = df["equity_eq_weight"].loc[start_date:] / df.loc[start_date, "equity_eq_weight"]

    equity_strat.plot(label="Strategy", linewidth=1.8, color="black")
    equity_bh.plot(label="Equal-Weight Buy & Hold", linestyle="--", color="gray")

    plt.ylabel("Equity (normalised to 1 at start)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # stacked weights
    assets_plot = ["SPY", "GLD", "TLT"]
    weight_cols = []
    col_map = {}
    for a in assets_plot:
        if f"w_{a}" in df.columns:
            col = f"w_{a}"
        elif a in df.columns:
            col = a
        else:
            found = next((c for c in df.columns if a in c), None)
            col = found
        if col:
            weight_cols.append(col)
            col_map[col] = a

    if not weight_cols:
        raise RuntimeError("No weight columns found for SPY/GLD/TLT in df.")


    grey_shades = [ "#7f7f7f", "#3f3f3f", "black"]  # light→dark without white/black
    custom_greys = mcolors.LinearSegmentedColormap.from_list("custom_greys", grey_shades)

    df_weights = df.loc[start_date:, weight_cols].rename(columns=col_map).abs()
    plt.figure(figsize=(11, 5))
    ax = df_weights.plot.area(ax=plt.gca(), stacked=True, alpha=1,
                              cmap=custom_greys, linewidth=0.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Asset", loc="lower right")
    plt.tight_layout()
    plt.show()

    # per-asset plots
    fig, axes = plt.subplots(len(assets_plot), 1, figsize=(11, 8), sharex=True)
    if len(assets_plot) == 1:
        axes = [axes]

    for ax, asset in zip(axes, assets_plot):
        df_asset = df.loc[start_date:]

        log_ret_full = returns_log_SPY_GLD_TLT[asset].reindex(df.index).fillna(0)
        log_ret = log_ret_full.loc[df_asset.index]
        cum_ret = np.exp(log_ret.cumsum())
        cum_ret = cum_ret / cum_ret.iloc[0]

        ax.plot(df_asset.index, cum_ret, color="black",
                label=f"{asset} price", lw=0.9)

        w_col = f"w_{asset}"
        if w_col in df_asset.columns:
            w = df_asset[w_col].fillna(0)
            pos = (w.abs() > 1e-8).astype(int)
            prev_pos = pos.shift(1)
            reb = df_asset["rebalance_flag"].fillna(False)

            buy_mask  = reb & prev_pos.notna() & (prev_pos == 0) & (pos == 1)
            sell_mask = reb & prev_pos.notna() & (prev_pos == 1) & (pos == 0)

            ax.scatter(
                df_asset.index[buy_mask],
                cum_ret[buy_mask],
                marker="^",
                color="green",
                s=45,
                label="Enter position",
                zorder=3,
                alpha=0.8,
            )
            ax.scatter(
                df_asset.index[sell_mask],
                cum_ret[sell_mask],
                marker="v",
                color="red",
                s=45,
                label="Exit position",
                zorder=3,
                alpha=0.8,
            )

        ax.set_ylabel("Equity")
        ax.set_title(asset)
        ax.grid(alpha=0.2)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()

# block stats + table
block_stats_SPY_GLD_TLT = df_SPY_GLD_TLT.attrs["block_stats"]

strategy_stats = {
    k: v for k, v in stats_SPY_GLD_TLT.items()
    if not isinstance(v, dict)
}

all_stats = {
    "Strategy":   strategy_stats,
    "EqWeight_BH": stats_SPY_GLD_TLT["eq_weight_bh"],
    "SPY_BH":      stats_SPY_GLD_TLT["SPY_bh"],
    "GLD_BH":      stats_SPY_GLD_TLT["GLD_bh"],
    "TLT_BH":      stats_SPY_GLD_TLT["TLT_bh"],
}

stats_table_SPY_GLD_TLT = pd.DataFrame(all_stats)
stats_table_SPY_GLD_TLT


