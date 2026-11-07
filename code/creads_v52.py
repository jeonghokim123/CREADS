#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CREADS v52 — TWO-LEVEL REGIME PIPELINE (country clustering added).

Flow (v52, per revision plan):
  Step 1  Geospatial data construction          (v51 protocol, unchanged)
  Step 2  Optimal country clustering            NEW : country geospatial features
                                                -> GMM -> BIC-based K*
  Step 3  Regional GMM within country clusters  NEW : spatial-only (circularity-
                                                free) regional types per cluster
  Step 4  Regime-specific FPCA                  per (country cluster x regional type)
  Step 5  Heterogeneous HDI prediction          Pass-1 pooled ensemble + Pass-2
                                                cell-specific FPC correction,
                                                fold-safe 5-fold CV ladder
  Step 6  Population-weighted targeting         CREADS vs pooled vs administrative
  Step 7  Context-sensitive policy scenarios    dose-response grid by regime cell

Every number of the v52 manuscript comes from THIS script (single execution,
seed 2024/42). The regime inputs contain NO predicted HDI, so the v52 typology
is circularity-free by construction.
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import creads_core as cc
import creads_core_improved as ci

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, r2_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "outputs_v52")
CSV = os.path.join(OUT, "csv_exports")
RTB = os.path.join(OUT, "result_tables")
for p in (OUT, CSV, RTB):
    os.makedirs(p, exist_ok=True)
cc.BASE = BASE
cc.DATA_DIR = os.path.join(BASE, "data")
cc.FILE_98 = os.path.join(cc.DATA_DIR, "MATCHED_340REGIONS_WITH_GDL_2023.xlsx")
cc.FILE_300 = os.path.join(cc.DATA_DIR, "COMPLETE_584REGIONS_WITH_CONNECTIVITY_2023.xlsx")

SEED = 2024
K_GRID_COUNTRY = [2, 3, 4, 5, 6]
K_GRID_REGIONAL = [2, 3]
TYPE_NAMES = {2: ["Sparse-Interior", "Dense-Core"],
              3: ["Sparse-Interior", "Transitional-Access", "Dense-Core"]}
CLUSTER_NAMES = {2: ["Low-Luminosity Cluster", "High-Luminosity Cluster"],
                 3: ["Low-Luminosity Cluster", "Middle-Luminosity Cluster", "High-Luminosity Cluster"],
                 4: ["Cluster 1 (lowest)", "Cluster 2", "Cluster 3", "Cluster 4 (highest)"],
                 5: ["Cluster 1 (lowest)", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5 (highest)"],
                 6: ["Cluster 1 (lowest)", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5", "Cluster 6 (highest)"]}
CLUSTER_SHORT = {2: ["C1", "C2"], 3: ["C1", "C2", "C3"],
                 4: ["C1", "C2", "C3", "C4"], 5: ["C1", "C2", "C3", "C4", "C5"],
                 6: ["C1", "C2", "C3", "C4", "C5", "C6"]}

# country-level geospatial features (Step 2) — NO welfare information
C_FEATS = ["nl", "popd", "urban", "border", "cap", "air", "smod", "tree", "nlgrad"]
C_FEAT_LABEL = {"nl": "log mean VIIRS radiance", "popd": "log mean population density",
                "urban": "mean urban ratio (25 km)", "border": "log mean border distance",
                "cap": "log mean distance to capital", "air": "log mean airport distance",
                "smod": "mean SMOD urban share", "tree": "mean tree-cover share",
                "nlgrad": "mean 1 km / 50 km radiance gradient"}

# regional spatial-only features (Step 3) — the v51 circularity-free set
GMM_SPATIAL_FEATS = ["nightlight", "border_distance", "pop_density",
                     "log_dist_capital", "log_dist_airport",
                     "urban_ratio", "smod_urban"]

N = {"meta": {"seed": SEED, "pipeline": "creads_v52.py"}}


def log(msg):
    print(msg, flush=True)


# ======================================================================
# Steps 1 — load + features
# ======================================================================
def step1():
    df98, df300 = cc.step1_load(verbose=False)
    df98, df300, _ = ci.step2_features_improved(df98, df300, verbose=False)
    return df98, df300


# ======================================================================
# Step 2 — optimal country clustering (GMM, BIC-selected K*)
# ======================================================================
def step2_country_clusters(df300):
    g = df300.groupby("ISO3")
    ctab = pd.DataFrame({
        "nl": np.log1p(g["viirs_mean"].mean().clip(lower=0)),
        "popd": np.log1p(g["pop_density"].mean().clip(lower=0)),
        "urban": g["urban_ratio"].mean(),
        "border": np.log1p(g["border_distance"].mean().clip(lower=0)),
        "cap": np.log1p(g["dist_to_capital_km"].mean().clip(lower=0)),
        "air": np.log1p(g["dist_to_nearest_airport_km"].mean().clip(lower=0)),
        "smod": g["smod_urban_ratio"].mean(),
        "tree": g["tree_cover"].mean(),
        "nlgrad": g["nl_ratio"].mean(),
    })
    X = ctab[C_FEATS].values.astype(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xsc = (X - mu) / sd

    rows, fits = [], {}
    for K in K_GRID_COUNTRY:
        gm = GaussianMixture(n_components=K, covariance_type="full",
                             random_state=SEED, n_init=20).fit(Xsc)
        lab = gm.predict(Xsc)
        sil = silhouette_score(Xsc, lab) if K > 1 else np.nan
        rows.append(dict(K=K, BIC=gm.bic(Xsc), AIC=gm.aic(Xsc), silhouette=sil))
        fits[K] = (gm, lab)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(CSV, "country_gmm_k_sensitivity.csv"), index=False)

    # selection: argmin BIC, subject to a pre-stated degenerate-partition guard
    # (every country cluster must contain at least 3 countries; with fewer, the
    #  cluster-level covariance cannot be estimated reliably). The guard and the
    # full sweep are reported in the manuscript.
    bic_argmin = int(sweep.loc[sweep.BIC.idxmin(), "K"])
    min_size = {K: int(pd.Series(fits[K][1]).value_counts().min()) for K in K_GRID_COUNTRY}
    sizes_by_K = {K: sorted(pd.Series(fits[K][1]).value_counts().tolist(), reverse=True)
                  for K in K_GRID_COUNTRY}
    Kstar = bic_argmin
    for K in sweep.sort_values("BIC")["K"].astype(int):
        if min_size[K] >= 3:
            Kstar = int(K)
            break
    guard_applied = (Kstar != bic_argmin)
    gm, raw = fits[Kstar]

    # relabel clusters by ascending country mean VIIRS
    order = pd.Series(ctab["nl"].values).groupby(raw).mean().sort_values().index.tolist()
    remap = {o: i + 1 for i, o in enumerate(order)}          # C1..CK*
    ctab["cluster"] = [remap[r] for r in raw]
    ctab["cluster_label"] = ctab["cluster"].map(
        dict(zip(range(1, Kstar + 1), CLUSTER_NAMES[Kstar])))
    N["country_clustering"] = {
        "K_star": Kstar,
        "bic_argmin": bic_argmin,
        "guard_applied": bool(guard_applied),
        "min_cluster_countries_by_K": {str(k): v for k, v in min_size.items()},
        "cluster_sizes_by_K": {str(k): v for k, v in sizes_by_K.items()},
        "bic": {str(int(r.K)): round(float(r.BIC), 2) for r in sweep.itertuples()},
        "aic": {str(int(r.K)): round(float(r.AIC), 2) for r in sweep.itertuples()},
        "silhouette": {str(int(r.K)): round(float(r.silhouette), 4) for r in sweep.itertuples()},
        "silhouette_at_Kstar": round(float(sweep.loc[sweep.K == Kstar, "silhouette"].iloc[0]), 4),
        "n_countries": int(len(ctab)),
        "sizes": {str(k): int(v) for k, v in ctab["cluster"].value_counts().sort_index().items()},
    }
    log(f"  [Step 2] country GMM K* = {Kstar} (BIC) | sizes = "
        f"{N['country_clustering']['sizes']}")
    return ctab, Kstar


# ======================================================================
# Step 3 — regional GMM within each country cluster (spatial-only)
# ======================================================================
def step3_regional_regimes(df300, ctab, Kstar):
    df300 = df300.merge(ctab[["cluster", "cluster_label"]].rename(
        columns={"cluster": "country_cluster"}), left_on="ISO3", right_index=True, how="left")
    feats = [f for f in GMM_SPATIAL_FEATS if f in df300.columns]
    sweep_rows, cell_tables = [], []
    for c in range(1, Kstar + 1):
        sub = df300[df300.country_cluster == c]
        X = pd.DataFrame(SimpleImputer_pd(sub[feats]), columns=feats).values.astype(float)
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xsc = (X - mu) / sd
        cands = []
        for K in K_GRID_REGIONAL:
            gm = GaussianMixture(n_components=K, covariance_type="full",
                                 random_state=SEED, n_init=20).fit(Xsc)
            lab = gm.predict(Xsc)
            sil = silhouette_score(Xsc, lab)
            bic = gm.bic(Xsc)
            min_cell = int(pd.Series(lab).value_counts().min())
            sweep_rows.append(dict(cluster=c, K=K, BIC=bic, AIC=gm.aic(Xsc),
                                   silhouette=sil, n=len(sub), min_cell=min_cell))
            cands.append((bic, K, lab, gm, sil, min_cell))
        feas = [t for t in cands if t[5] >= 8]
        bic, Kc, lab, gm, sil, min_cell = sorted(feas or cands, key=lambda t: t[0])[0]
        names = TYPE_NAMES[Kc]
        # relabel by ascending VIIRS
        order = pd.Series(sub["nightlight"].values).groupby(lab).mean().sort_values().index.tolist()
        remap = {o: i for i, o in enumerate(order)}
        typ = np.array([remap[r] for r in lab])
        df300.loc[sub.index, "regional_type"] = typ
        df300.loc[sub.index, "regime_label"] = [
            f"{CLUSTER_SHORT[Kstar][c-1]}-{names[t]}" for t in typ]
        for t in range(Kc):
            m = typ == t
            cell_tables.append(dict(country_cluster=c, cluster_label=CLUSTER_NAMES[Kstar][c - 1],
                                    regional_type=t, type_name=names[t],
                                    regime_label=f"{CLUSTER_SHORT[Kstar][c-1]}-{names[t]}",
                                    n_regions=int(m.sum())))
        log(f"  [Step 3] cluster C{c} (n={len(sub)}): K_c={Kc} (BIC), silhouette={sil:.4f}, "
            f"sizes={[int((typ==t).sum()) for t in range(Kc)]}")
    pd.DataFrame(sweep_rows).to_csv(os.path.join(CSV, "regional_gmm_k_sensitivity.csv"), index=False)
    df300["regime_cell"] = df300["regime_label"]
    return df300, feats


def SimpleImputer_pd(sub_feats):
    from sklearn.impute import SimpleImputer
    return SimpleImputer(strategy="median").fit_transform(sub_feats.values.astype(float))


# ======================================================================
# shared model helpers (v51 protocol, pinned)
# ======================================================================
def ef(Xtr, ytr, rf_n=300, gb_n=200):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    imp = SimpleImputer(strategy="median").fit(Xtr)
    Xt = imp.transform(Xtr)
    s = StandardScaler().fit(Xt)
    Xs = s.transform(Xt)
    rf = RandomForestRegressor(n_estimators=rf_n, max_depth=12, min_samples_leaf=2,
                               n_jobs=-1, random_state=SEED).fit(Xt, ytr)
    gb = GradientBoostingRegressor(n_estimators=gb_n, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, random_state=SEED).fit(Xt, ytr)
    rd = Ridge(alpha=1.0).fit(Xs, ytr)
    w = np.clip([r2_score(ytr, rf.predict(Xt)), r2_score(ytr, gb.predict(Xt)),
                 r2_score(ytr, rd.predict(Xs))], 0, None)
    w = w / w.sum() if w.sum() > 0 else np.array([1, 1, 1]) / 3
    return imp, s, rf, gb, rd, w


def ep(M, A_):
    imp, s, rf, gb, rd, w = M
    A = imp.transform(A_)
    B = s.transform(A)
    return w[0] * rf.predict(A) + w[1] * gb.predict(A) + w[2] * rd.predict(B)


def wr2(yy, pp, gg):
    ym = pd.Series(yy).groupby(gg).transform("mean").values
    pm = pd.Series(pp).groupby(gg).transform("mean").values
    return r2_score(yy - ym, pp - pm)


def profiles(df):
    pc = [f"viirs_mean_{k}km" for k in [1, 5, 10, 25, 50]]
    raw = np.nan_to_num(df[pc].values.astype(float), nan=0.0)
    L = np.log1p(np.clip(raw, 0, None))
    return L - L.mean(1, keepdims=True)


def cell_bases(P, idx, cells, order_cells, n=3):
    """FPCA basis per regime cell; fallback to cluster basis then pooled."""
    B = {}
    cl_of = CELLS_CLUSTER  # global map cell -> cluster
    for cell in order_cells:
        m = idx[cells[idx] == cell]
        if len(m) < 12:
            m = idx[cells[idx] == cell] if False else None
        if m is not None and len(m) >= 12:
            mu = P[m].mean(0); sd = P[m].std(0) + 1e-9
            B[cell] = (mu, sd, np.linalg.svd((P[m] - mu) / sd, full_matrices=False)[2][:n],
                       "cell")
    # cluster fallback for small cells
    for cell in order_cells:
        if cell in B:
            continue
        cm = idx[np.array([cells[i] == cl_of[cell] for i in idx])]
        if len(cm) >= 12:
            mu = P[cm].mean(0); sd = P[cm].std(0) + 1e-9
            B[cell] = (mu, sd, np.linalg.svd((P[cm] - mu) / sd, full_matrices=False)[2][:n],
                       "cluster")
    # pooled fallback
    for cell in order_cells:
        if cell in B:
            continue
        mu = P[idx].mean(0); sd = P[idx].std(0) + 1e-9
        B[cell] = (mu, sd, np.linalg.svd((P[idx] - mu) / sd, full_matrices=False)[2][:n], "pooled")
    return B


def cell_scores(B, P, idx, cells, n=3):
    o = np.zeros((len(idx), n))
    for cell, (mu, sd, V, _) in B.items():
        m = cells[idx] == cell
        if m.sum() == 0:
            continue
        o[m] = ((P[idx][m] - mu) / sd) @ V.T
    return o


# ======================================================================
# Steps 4+5 — fold-safe CV ladder with cell FPCA
# ======================================================================
def step45_cv_ladder(df98, cells_lab, order_cells, FE):
    v = df98["HDI"].notna().values
    X = df98.loc[v, FE].values.astype(float)
    y = df98.loc[v, "HDI"].values.astype(float)
    g = df98.loc[v, "ISO3"].values
    cells = cells_lab
    SH = profiles(df98.loc[v])
    idx_all = np.arange(len(y))

    from sklearn.model_selection import KFold
    OUT5 = KFold(5, shuffle=True, random_state=42)

    def cv_run(mode):
        oof = np.zeros(len(y)); folds = []
        for tr, te in OUT5.split(X):
            ctr = pd.Series(g).iloc[tr].to_frame("ISO3").assign(H=y[tr]).groupby("ISO3")["H"].mean()
            gm = float(y[tr].mean())
            ytr = y[tr] - pd.Series(g[tr]).map(ctr).fillna(gm).values
            add = pd.Series(g[te]).map(ctr).fillna(gm).values
            B = cell_bases(SH, tr, cells, order_cells) if mode in ("fpc", "pass2") else None
            if mode == "fpc":
                Xtr = np.column_stack([X[tr], cell_scores(B, SH, tr, cells)])
                Xte = np.column_stack([X[te], cell_scores(B, SH, te, cells)])
            elif mode == "regind":
                D = pd.get_dummies(pd.Series(cells), prefix="R").reindex(
                    columns=pd.get_dummies(pd.Series(cells), prefix="R").columns
                ).values.astype(float)
                Xtr = np.column_stack([X[tr], D[tr]]); Xte = np.column_stack([X[te], D[te]])
            else:
                Xtr, Xte = X[tr], X[te]
            M = ef(Xtr, ytr)
            if mode == "hard":
                p = np.zeros(len(te))
                for cell in order_cells:
                    m1 = cells[tr] == cell; m2 = cells[te] == cell
                    if m2.sum() == 0:
                        continue
                    if m1.sum() < 15:
                        p[m2] = ep(M, X[te][m2])
                    else:
                        Mk = ef(X[tr][m1], ytr[m1]); p[m2] = ep(Mk, X[te][m2])
                oof[te] = p + add
            elif mode == "pass2":
                b_tr = ep(M, Xtr); r = ytr - b_tr
                adj = np.zeros(len(te))
                for cell in order_cells:
                    m1 = cells[tr] == cell; m2 = cells[te] == cell
                    if m1.sum() < 12 or m2.sum() == 0:
                        continue
                    F1tr = cell_scores(B, SH, tr, cells)[:, 0][m1]
                    D = np.column_stack([F1tr, F1tr ** 2])
                    # inner 4-fold lambda selection
                    from sklearn.linear_model import Ridge
                    ikf = KFold(4, shuffle=True, random_state=7)
                    best_l, best_sse = 0.0, np.inf
                    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
                        sse = 0.0
                        for itr, ite in ikf.split(D):
                            if len(ite) == 0:
                                continue
                            rd = Ridge(alpha=2.0).fit(D[itr], r[m1][itr])
                            sse += ((r[m1][ite] - rd.predict(D[ite])) ** 2).sum()
                        if sse < best_sse:
                            best_sse, best_l = sse, lam
                    rd = Ridge(alpha=2.0).fit(D, r[m1])
                    F1te = cell_scores(B, SH, te, cells)[:, 0][m2]
                    Dt = np.column_stack([F1te, F1te ** 2])
                    adj[m2] = best_l * rd.predict(Dt)
                oof[te] = ep(M, Xte) + adj + add
            else:
                oof[te] = ep(M, Xte) + add
            folds.append(float(wr2(y[te], oof[te], g[te])))
        return oof, folds

    N["ladder"] = {}
    oofs = {}
    for tag, mode in [("pooled", "pooled"), ("plus_fpc", "fpc"),
                      ("plus_regind", "regind"), ("hard", "hard"), ("creads", "pass2")]:
        o, f = cv_run(mode)
        oofs[tag] = o
        N["ladder"][tag] = dict(overall=round(float(r2_score(y, o)), 4),
                                within=round(float(wr2(y, o, g)), 4),
                                rmse=round(float(np.sqrt(((y - o) ** 2).mean())), 4),
                                fold_sd=round(float(np.std(f)), 4))
        log(f"  [Step 5] ladder {tag:11s} overall={N['ladder'][tag]['overall']:.4f} "
            f"within={N['ladder'][tag]['within']:.4f}")

    # in-sample full model + weights
    ctr = pd.Series(g).to_frame("ISO3").assign(H=y).groupby("ISO3")["H"].mean()
    ydm = y - pd.Series(g).map(ctr).values
    M = ef(X, ydm)
    yis = ep(M, X) + pd.Series(g).map(ctr).values
    N["insample"] = dict(r2=round(float(r2_score(y, yis)), 4),
                         rmse=round(float(np.sqrt(((y - yis) ** 2).mean())), 4),
                         weights=dict(rf=round(float(M[5][0]), 3), gb=round(float(M[5][1]), 3),
                                      ridge=round(float(M[5][2]), 3)))
    return X, y, g, cells, SH, oofs, M, ctr


# ======================================================================
# leave-one-country-out
# ======================================================================
def step5_loco(X, y, g, cells, SH, order_cells):
    per = {}; o_l = np.zeros(len(y)); o_f = np.zeros(len(y))
    for c in sorted(set(g)):
        te = np.where(g == c)[0]; tr = np.where(g != c)[0]
        gm = float(y[tr].mean())
        M = ef(X[tr], y[tr] - gm, rf_n=150, gb_n=100)
        o_l[te] = ep(M, X[te]) + gm
        # FPC-augmented variant
        B = cell_bases(SH, tr, cells, order_cells)
        Ftr = cell_scores(B, SH, tr, cells); Fte = cell_scores(B, SH, te, cells)
        Mf = ef(np.column_stack([X[tr], Ftr]), y[tr] - gm, rf_n=150, gb_n=100)
        o_f[te] = ep(Mf, np.column_stack([X[te], Fte])) + gm
        if len(te) >= 5:
            per[c] = float(r2_score(y[te], o_l[te]))
    N["loco"] = dict(overall=round(float(r2_score(y, o_l)), 4),
                     rmse=round(float(np.sqrt(((y - o_l) ** 2).mean())), 4),
                     within=round(float(wr2(y, o_l, g)), 4),
                     median=round(float(np.median(list(per.values()))), 3),
                     q1=round(float(np.percentile(list(per.values()), 25)), 3),
                     q3=round(float(np.percentile(list(per.values()), 75)), 3),
                     n_positive=int(sum(1 for z in per.values() if z > 0)),
                     n_countries=len(per))
    N["loco_fpc"] = dict(overall=round(float(r2_score(y, o_f)), 4),
                         rmse=round(float(np.sqrt(((y - o_f) ** 2).mean())), 4),
                         within=round(float(wr2(y, o_f, g)), 4))
    log(f"  [Step 5] LOCO overall={N['loco']['overall']} within={N['loco']['within']}")


# ======================================================================
# predict 584 + Step 6 targeting
# ======================================================================
def step6_predict_target(df300, df98, M, ctr, global_mean, FE, order_cells):
    Xr = np.zeros((len(df300), len(FE)))
    for i, f in enumerate(FE):
        if f in df300.columns:
            col = df300[f]
            Xr[:, i] = col.fillna(col.median() if col.notna().any() else 0.0).values
        else:
            Xr[:, i] = 0.0
    pred = np.clip(ep(M, Xr) + df300["ISO3"].map(ctr).fillna(global_mean).values, 0.30, 0.99)
    df300["HDI_predicted"] = pred

    # ---- population-weighted targeting on the labelled sample ----
    v = df98["HDI"].notna().values
    y = df98.loc[v, "HDI"].values
    rid = df98.loc[v, "region_id"].values
    lab = df98.loc[v, ["region_id", "ISO3", "pop_sum"]].copy()
    cmap = dict(zip(oofs_lookup["rid"], oofs_lookup["creads"]))
    pmap = dict(zip(oofs_lookup["rid"], oofs_lookup["pooled"]))
    lab["creads"] = lab.region_id.map(cmap).values
    lab["pooled"] = lab.region_id.map(pmap).values
    lab["obs"] = y
    lab["admin"] = lab.groupby("ISO3")["obs"].transform("mean")
    lab["pop"] = lab.pop_sum

    def bottom(df, col, frac=0.20):
        s = df.sort_values(col, kind="mergesort")
        c = s["pop"].cumsum() / s["pop"].sum()
        return set(s[c.shift(fill_value=0) < frac].region_id)

    poor = bottom(lab, "obs")
    ppop = lab[lab.region_id.isin(poor)]["pop"].sum()
    base = ppop / lab["pop"].sum()

    def score(sel):
        cap = poor & sel
        cp = lab[lab.region_id.isin(cap)]["pop"].sum()
        sp = lab[lab.region_id.isin(sel)]["pop"].sum()
        return dict(n=len(sel), recall=round(cp / ppop * 100, 1),
                    precision=round(cp / sp * 100, 1), lift=round(cp / sp / base, 2))

    N["targeting"] = {}
    for nm, col in [("creads", "creads"), ("pooled", "pooled")]:
        N["targeting"][nm] = score(bottom(lab, col))
    rng = np.random.default_rng(0)
    acc = [score(bottom(lab.assign(k=lab.admin + rng.random(len(lab)) * 1e-7), "k")) for _ in range(400)]
    N["targeting"]["admin"] = dict(
        n=int(round(np.mean([a["n"] for a in acc]))),
        recall=round(float(np.mean([a["recall"] for a in acc])), 1),
        precision=round(float(np.mean([a["precision"] for a in acc])), 1),
        lift=round(float(np.mean([a["lift"] for a in acc])), 2),
        sd_recall=round(float(np.std([a["recall"] for a in acc])), 1),
        sd_precision=round(float(np.std([a["precision"] for a in acc])), 1))
    N["targeting"]["base_rate"] = round(base * 100, 1)
    N["targeting"]["n_poor"] = len(poor)

    # counts on the 584 inventory
    d = df300.copy()
    d["adm"] = d.groupby("ISO3")["HDI_predicted"].transform("mean")
    def count(sel_col, tie=None, tie_desc=False):
        by = [sel_col] + ([tie] if tie else [])
        asc = [True] + ([not tie_desc] if tie else [])
        s = d.sort_values(by, ascending=asc, kind="mergesort")
        c = s["pop_sum"].cumsum() / s["pop_sum"].sum()
        return int((c.shift(fill_value=0) < 0.20).sum())
    N["targeting"]["n_584_budget"] = dict(creads=count("HDI_predicted"),
                                          admin=count("adm"),
                                          admin_poporder=count("adm", "pop_sum", tie_desc=True))
    # partial ROC diagnostic (sigma=0.10 latent-poor protocol)
    from scipy.stats import norm
    lm = lab.copy().sort_values("obs", kind="mergesort")
    cw = lm["pop"].cumsum() / lm["pop"].sum()
    tau = float(lm.loc[cw >= 0.20, "obs"].iloc[0])
    mu = lm["obs"].values
    pop = lm["pop"].values
    poor_prob = norm.cdf((tau - mu) / 0.10)
    shares = np.linspace(0.02, 0.80, 60)

    def partial_auc(score_col):
        order = np.argsort(lab[score_col].values)          # ascending = poorest first
        tpr, fpr = [], []
        for s in shares:
            k = max(1, int(round(s * len(mu))))
            sel = order[:k]
            tp = pop[sel] * poor_prob[sel]; fp = pop[sel] * (1 - poor_prob[sel])
            tpr.append(tp.sum() / (pop * poor_prob).sum())
            fpr.append(fp.sum() / (pop * (1 - poor_prob)).sum())
        tpr, fpr = np.array(tpr), np.array(fpr)
        return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))

    N["targeting"]["roc_predicted"] = round(partial_auc("creads"), 4)
    N["targeting"]["roc_admin"] = round(partial_auc("admin"), 4)
    log(f"  [Step 6] targeting creads={N['targeting']['creads']} "
        f"admin={N['targeting']['admin']} n584={N['targeting']['n_584_budget']}")


# ======================================================================
# Step 7 — policy scenarios per regime cell
# ======================================================================
FE5 = ["viirs_mean", "dist_to_capital_km", "pop_density", "urban_ratio_25km",
       "dist_to_nearest_airport_km"]


def prof584(df):
    P = np.log(np.clip(df[[f"viirs_mean_{k}km" for k in [1, 5, 10, 25, 50]]].values, 1e-3, None))
    return P - P.mean(1, keepdims=True)


def mk(df, fpc=True, inter=True):
    L = [np.log1p(df[c].clip(lower=0).values) for c in FE5]
    cols = [np.ones(len(df))] + L
    if inter:
        cols += [L[0] * L[1], L[0] * L[2]]
    if fpc:
        cols += [prof584(df)[:, 0]]
    return np.column_stack(cols)


def step7_policy(df300, df98, cells_lab, order_cells):
    v = df98["HDI"].notna().values
    lab5 = df98.loc[v].copy()
    lab5["regime"] = cells_lab
    lab5 = lab5.dropna(subset=FE5 + ["regime"]).copy()
    lab5 = lab5[np.isfinite(lab5[FE5].values).all(1)]
    lab5["obs"] = lab5["HDI"].values
    d3 = df300.dropna(subset=FE5 + ["regime_label"]).copy()
    d3 = d3[np.isfinite(d3[FE5].values).all(1)].rename(columns={"regime_label": "regime"})

    N["policy"] = {"spec_r2": {}, "dose": {}, "access": {}, "frontier": {}}
    for tag, (f_, i_) in {"A": (False, False), "B": (True, False),
                          "C": (False, True), "D": (True, True)}.items():
        Xl = mk(lab5, f_, i_)
        mods = {}
        for k in order_cells:
            m = (lab5.regime == k).values
            if m.sum() < 15:                     # fallback: pooled coefficients
                m = np.ones(len(lab5), dtype=bool)
            b = np.linalg.lstsq(Xl[m], lab5.obs.values[m], rcond=None)[0]
            mods[k] = b
            mm = (lab5.regime == k).values
            if mm.sum() > 0:
                N["policy"]["spec_r2"].setdefault(tag, {})[k] = round(float(
                    1 - ((lab5.obs.values[mm] - Xl[mm] @ b) ** 2).sum() /
                    ((lab5.obs.values[mm] - lab5.obs.values[mm].mean()) ** 2).sum()), 3)
        A = mk(d3, f_, i_)

        def gain(floor=None, cut=None):
            gg = np.zeros(len(d3))
            for k, b in mods.items():
                m = (d3.regime == k).values
                if m.sum() == 0:
                    continue
                Ak = A[m].copy()
                if floor is not None:
                    Ak[:, 1] = np.log1p(np.maximum(d3.viirs_mean.values[m], floor))
                if cut is not None:
                    Ak[:, 2] = np.log1p(d3.dist_to_capital_km.values[m] * (1 - cut))
                gg[m] = Ak @ b - A[m] @ b
            return gg

        if tag == "D":
            for fl in [0.5, 1.0, 1.5, 2.5, 4.0]:
                gg = gain(floor=fl); row = {}
                for k in order_cells:
                    m = (d3.regime == k).values
                    if m.sum() == 0:
                        continue
                    row[k] = round(float(np.average(gg[m], weights=d3.pop_sum.values[m])), 4)
                N["policy"]["dose"][str(fl)] = row
            for cut in [0.10, 0.25, 0.50]:
                gg = gain(cut=cut); row = {}
                for k in order_cells:
                    m = (d3.regime == k).values
                    if m.sum() == 0:
                        continue
                    row[k] = round(float(np.average(gg[m], weights=d3.pop_sum.values[m])), 4)
                N["policy"]["access"][str(cut)] = row
        d3["G"] = np.clip(gain(floor=1.5), 0, None)
        d3["adm"] = d3.groupby("ISO3")["HDI_predicted"].transform("mean")
        rows = []
        for K in list(range(20, 241, 10)) + [92]:
            lv = d3.sort_values("HDI_predicted", kind="mergesort").head(K)
            ad = d3.sort_values(["adm", "HDI_predicted"], kind="mergesort").head(K)
            wf = d3.sort_values("G", ascending=False, kind="mergesort").head(K)
            rows.append(dict(K=K, level=round(float(lv.G.sum()), 3),
                             admin=round(float(ad.G.sum()), 3),
                             welfare=round(float(wf.G.sum()), 3)))
        N["policy"]["frontier"][tag] = rows
        if tag == "D":
            N["policy"]["composition_92"] = d3.sort_values(
                "G", ascending=False, kind="mergesort").head(92).regime.value_counts().to_dict()
    log("  [Step 7] policy scenarios done")


# ======================================================================
# diagnostics — separation, stability, FPCA
# ======================================================================
CELLS_CLUSTER = {}


def step8_diagnostics(df300, df98, cells_lab, order_cells, Kstar, spatial_feats):
    v = df98["HDI"].notna().values
    # ---- partition separation on globally standardised spatial features ----
    feats = [f for f in spatial_feats if f in df300.columns]
    Xinv = pd.DataFrame(SimpleImputer_pd(df300[feats]), columns=feats).values.astype(float)
    Xsc = (Xinv - Xinv.mean(0)) / (Xinv.std(0) + 1e-9)
    part = df300["regime_label"].values
    sil_new = float(silhouette_score(Xsc, part))

    # single-level spatial-only GMM (K=3) on the same features
    gm3 = GaussianMixture(n_components=3, covariance_type="full", random_state=SEED,
                          n_init=20).fit(Xsc)
    sil_spatial_k3 = float(silhouette_score(Xsc, gm3.predict(Xsc)))

    # welfare-integrated alternative (recomputed fresh in this run)
    Xw = df300[["nightlight", "border_distance", "pop_density", "HDI_predicted"]].values.astype(float)
    Xw = np.nan_to_num(Xw, nan=np.nanmedian(Xw, axis=0))
    Xws = (Xw - Xw.mean(0)) / (Xw.std(0) + 1e-9)
    gmw = GaussianMixture(n_components=3, covariance_type="full", random_state=SEED,
                          n_init=20).fit(Xws)
    lw = gmw.predict(Xws)
    sil_welfare = float(silhouette_score(Xws, lw))

    iso = df300["ISO3"].astype(str).values
    nmi = float(normalized_mutual_info_score(iso, part))
    purity = float(pd.crosstab(part, iso).max(axis=1).sum() / len(part))

    N["partition"] = dict(silhouette_two_level=round(sil_new, 4),
                          silhouette_spatial_only_k3=round(sil_spatial_k3, 4),
                          silhouette_welfare_integrated_k3=round(sil_welfare, 4),
                          nmi_vs_country=round(nmi, 4), purity_vs_country=round(purity, 4),
                          n_cells=int(len(order_cells)))
    log(f"  [Diag] silhouette two-level={sil_new:.4f} spatial-K3={sil_spatial_k3:.4f} "
        f"welfare-integrated-K3={sil_welfare:.4f} NMI={nmi:.4f} purity={purity:.4f}")

    # ---- bootstrap stability of the two-level partition ----
    rng = np.random.default_rng(11)
    ct = df300.groupby("ISO3")  # country features were built on df300 groups
    ctab_rows = []
    g = df300.groupby("ISO3")
    C = pd.DataFrame({
        "nl": np.log1p(g["viirs_mean"].mean().clip(lower=0)),
        "popd": np.log1p(g["pop_density"].mean().clip(lower=0)),
        "urban": g["urban_ratio"].mean(),
        "border": np.log1p(g["border_distance"].mean().clip(lower=0)),
        "cap": np.log1p(g["dist_to_capital_km"].mean().clip(lower=0)),
        "air": np.log1p(g["dist_to_nearest_airport_km"].mean().clip(lower=0)),
        "smod": g["smod_urban_ratio"].mean(),
        "tree": g["tree_cover"].mean(),
        "nlgrad": g["nl_ratio"].mean()})
    Csc = (C[C_FEATS].values - C[C_FEATS].values.mean(0)) / (C[C_FEATS].values.std(0) + 1e-9)
    gmC = GaussianMixture(n_components=Kstar, covariance_type="full", random_state=SEED,
                          n_init=20).fit(Csc)
    point_c = gmC.predict(Csc)
    aris_c = []
    from sklearn.metrics import adjusted_rand_score
    for b in range(200):
        idxb = rng.integers(0, len(C), len(C))
        gmB = GaussianMixture(n_components=Kstar, covariance_type="full", random_state=SEED,
                              n_init=10).fit(Csc[idxb])
        uniq = np.unique(idxb)
        labB = gmB.predict(Csc[uniq])
        labP = point_c[uniq]
        # match bootstrap labels to point labels (Hungarian on contingency)
        K = Kstar
        cont = np.zeros((K, K))
        for a in range(K):
            for b2 in range(K):
                cont[a, b2] = np.sum((labB == a) & (labP == b2))
        ra, cb = linear_sum_assignment(-cont)
        remap = {int(a): int(b2) for a, b2 in zip(ra, cb)}
        labBm = np.array([remap.get(int(x), -1) for x in labB])
        aris_c.append(adjusted_rand_score(labP, labBm))
    # regional level: within each fixed country cluster
    aris_r = []
    for c in range(1, Kstar + 1):
        sub = df300[df300.country_cluster == c]
        Xs = pd.DataFrame(SimpleImputer_pd(sub[feats]), columns=feats).values.astype(float)
        mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
        Xss = (Xs - mu) / sd
        Kc = int(sub["regional_type"].max()) + 1
        gmR = GaussianMixture(n_components=Kc, covariance_type="full", random_state=SEED,
                              n_init=20).fit(Xss)
        point_r = gmR.predict(Xss)
        for b in range(200):
            idxb = rng.integers(0, len(Xss), len(Xss))
            gmB = GaussianMixture(n_components=Kc, covariance_type="full", random_state=SEED,
                                  n_init=10).fit(Xss[idxb])
            uniq = np.unique(idxb)
            labB = gmB.predict(Xss[uniq])
            labP = point_r[uniq]
            cont = np.zeros((Kc, Kc))
            for a in range(Kc):
                for b2 in range(Kc):
                    cont[a, b2] = np.sum((labB == a) & (labP == b2))
            ra, cb = linear_sum_assignment(-cont)
            remap = {int(a): int(b2) for a, b2 in zip(ra, cb)}
            labBm = np.array([remap.get(int(x), -1) for x in labB])
            aris_r.append(adjusted_rand_score(labP, labBm))
    N["partition"]["bootstrap"] = dict(
        country_median_ari=round(float(np.median(aris_c)), 3),
        country_p05_ari=round(float(np.percentile(aris_c, 5)), 3),
        regional_median_ari=round(float(np.median(aris_r)), 3),
        regional_p05_ari=round(float(np.percentile(aris_r, 5)), 3))

    # ---- FPCA: variance shares + bootstrap cosine per cell ----
    SH = profiles(df98.loc[v])
    cells_v = np.array(cells_lab)
    fp = {}
    for cell in order_cells:
        m = cells_v == cell
        if m.sum() < 12:
            continue
        P = SH[m]
        mu, sd = P.mean(0), P.std(0) + 1e-9
        V = np.linalg.svd((P - mu) / sd, full_matrices=False)[2]
        evr = np.linalg.svd((P - mu) / sd, full_matrices=False)[1] ** 2
        evr = evr / evr.sum()
        rng2 = np.random.default_rng(23)
        cos = []
        for b in range(200):
            ib = rng2.integers(0, len(P), len(P))
            mub, sdb = P[ib].mean(0), P[ib].std(0) + 1e-9
            Vb = np.linalg.svd((P[ib] - mub) / sdb, full_matrices=False)[2][0]
            cos.append(abs(float(V[0] @ Vb)))
        fp[cell] = dict(n=int(m.sum()),
                        pc1_pct=round(float(evr[0] * 100), 1),
                        cum3_pct=round(float(evr[:3].sum() * 100), 1),
                        cos1_median=round(float(np.median(cos)), 3),
                        cos1_p05=round(float(np.percentile(cos, 5)), 3))
    N["fpca_cells"] = fp

    # per-regime correlation of FPC1 with welfare residual (creads OOF)
    res = N["_y"] - N["_oofs"]["creads"]
    F1_full = cell_scores(cell_bases(SH, np.arange(len(cells_v)), cells_v, order_cells),
                          SH, np.arange(len(cells_v)), cells_v)[:, 0]
    N["fpca_welfare"] = {}
    for cell in order_cells:
        m = cells_v == cell
        if m.sum() < 15:
            continue
        N["fpca_welfare"][cell] = dict(
            corr=round(float(np.corrcoef(F1_full[m], res[m])[0, 1]), 3),
            var_share=round(float(np.var(F1_full[m] * np.polyfit(F1_full[m], res[m], 1)[0]) /
                                  np.var(res[m])), 3))

    # per-cell within-country R2 (hierarchical vs pooled)
    N["per_cell_r2"] = {}
    g = N["_g"]
    for cell in order_cells:
        m = cells_v == cell
        if m.sum() < 10:
            continue
        ym = pd.Series(N["_y"][m]).groupby(g[m]).transform("mean").values
        pm = pd.Series(N["_oofs"]["creads"][m]).groupby(g[m]).transform("mean").values
        pm2 = pd.Series(N["_oofs"]["pooled"][m]).groupby(g[m]).transform("mean").values
        yy = N["_y"][m] - ym
        N["per_cell_r2"][cell] = dict(creads=round(float(r2_score(yy, N["_oofs"]["creads"][m] - pm)), 3),
                                      pooled=round(float(r2_score(yy, N["_oofs"]["pooled"][m] - pm2)), 3))


# ======================================================================
# main
# ======================================================================
def main():
    log("=" * 72)
    log("  CREADS v52 — two-level (country x regional) regime pipeline")
    log("=" * 72)

    df98, df300 = step1()
    FE = [f for f in ci.FEATS_IMPROVED if f in df98.columns]

    # Step 2
    ctab, Kstar = step2_country_clusters(df300)
    ctab.reset_index().rename(columns={"index": "ISO3"}).to_csv(
        os.path.join(CSV, "country_cluster_membership.csv"), index=False)

    # Step 3
    df300, spatial_feats = step3_regional_regimes(df300, ctab, Kstar)
    order_cells = (df300.groupby("regime_label")
                   .agg(c=("country_cluster", "first"), t=("regional_type", "first"),
                        n=("region_id", "size"))
                   .sort_values(["c", "t"]).reset_index()["regime_label"].tolist())
    global CELLS_CLUSTER
    CELLS_CLUSTER = dict(zip(order_cells, df300.drop_duplicates("regime_label")
                             .set_index("regime_label").loc[order_cells, "country_cluster"]))

    # labelled cells
    df98 = df98.merge(df300[["rid", "country_cluster", "regional_type", "regime_label"]],
                      on="rid", how="left")
    v = df98["HDI"].notna().values
    cells_lab = df98.loc[v, "regime_label"].fillna("C1-Sparse-Interior").values
    # safety: fill unassigned with pooled label
    fallback = order_cells[0]
    cells_lab = np.array([c if c in order_cells else fallback for c in cells_lab])

    # Steps 4+5
    X, y, g, cells, SH, oofs, M, ctr = step45_cv_ladder(df98, cells_lab, order_cells, FE)
    N["_y"], N["_g"], N["_oofs"] = y, g, oofs
    global oofs_lookup
    rid_v = df98.loc[v, "region_id"].values
    oofs_lookup = pd.DataFrame({"rid": rid_v, "creads": oofs["creads"], "pooled": oofs["pooled"]})

    # LOCO
    step5_loco(X, y, g, cells, SH, order_cells)

    # Step 6 — predict 584 + targeting (fills HDI_predicted)
    step6_predict_target(df300, df98, M, ctr, float(y.mean()), FE, order_cells)

    # regime summary (584) — after prediction so HDI_predicted exists
    gsum = df300.groupby("regime_label").agg(
        N=("region_id", "size"), viirs=("viirs_mean", "mean"),
        popd=("pop_density", "mean"), hdi_pred=("HDI_predicted", "mean"),
        pop=("pop_sum", "sum")).reindex(order_cells)
    joined = df300[["rid", "regime_label"]].merge(
        df98.loc[df98["HDI"].notna(), ["rid", "HDI"]], on="rid", how="inner")
    gsum["hdi_obs"] = joined.groupby("regime_label")["HDI"].mean().reindex(order_cells)
    gsum["share_pct"] = (gsum["N"] / gsum["N"].sum() * 100).round(1)
    gsum["pop_share_pct"] = (gsum["pop"] / gsum["pop"].sum() * 100).round(1)
    gsum.round(4).to_csv(os.path.join(CSV, "regime_summary_statistics.csv"))
    log("  [Summary]\n" + gsum.round(3).to_string())

    # Step 7 — policy scenarios
    step7_policy(df300, df98, cells_lab, order_cells)

    # diagnostics
    step8_diagnostics(df300, df98, cells_lab, order_cells, Kstar, spatial_feats)

    # ---- save artefacts ----
    df300[["rid", "ISO3", "NAME_1", "country_cluster", "regime_label",
           "HDI_predicted", "pop_sum"]].rename(columns={"rid": "region_id"}).to_csv(
        os.path.join(CSV, "predicted_584_regions_with_cluster_regimes.csv"), index=False)
    df98.loc[v, ["region_id", "ISO3", "NAME_1", "HDI", "regime_label"]].to_csv(
        os.path.join(CSV, "regime_340regions.csv"), index=False)
    # targeting summary CSV for fig7
    pd.DataFrame({
        "metric": ["n_regions", "n_admin_groups", "admin_selected_regions",
                   "pred_selected_regions", "predicted_auc", "admin_auc"],
        "value": [len(df300), df300["ISO3"].nunique(),
                  N["targeting"]["n_584_budget"]["admin"],
                  N["targeting"]["n_584_budget"]["creads"],
                  N["targeting"]["roc_predicted"], N["targeting"]["roc_admin"]]
    }).to_csv(os.path.join(RTB, "table_spatial_targeting_summary.csv"), index=False)
    # quintile assignment + transition for fig8
    d = df300.copy()
    d["adm"] = d.groupby("ISO3")["HDI_predicted"].transform("mean")
    os.makedirs(os.path.join(OUT, "spatial_targeting"), exist_ok=True)
    # population-weighted quintiles: rank by cumulative population share, split in 5
    def assign_q(col, tie="HDI_predicted"):
        s = d.sort_values([col, tie], kind="mergesort")
        c = (s["pop_sum"].cumsum() / s["pop_sum"].sum())
        q = np.minimum((c * 5).fillna(0).astype(int) + 1, 5)
        return pd.Series(q.values, index=s.index).reindex(d.index)
    d["aq"] = assign_q("adm"); d["pq"] = assign_q("HDI_predicted")
    d[["rid", "ISO3", "NAME_1", "pop_sum", "aq", "pq"]].rename(
        columns={"rid": "region_id", "aq": "admin_quintile", "pq": "pred_quintile"}).to_csv(
        os.path.join(OUT, "spatial_targeting", "figure7_quintile_assignment_regions.csv"),
        index=False)
    comp = np.zeros((5, 5))
    for i in range(1, 6):
        for j in range(1, 6):
            comp[i - 1, j - 1] = d[(d.aq == i) & (d.pq == j)]["pop_sum"].sum()
    comp = comp / np.maximum(comp.sum(axis=1, keepdims=True), 1e-9) * 100
    pd.DataFrame(comp.round(1), index=[f"Q{i}" for i in range(1, 6)],
                 columns=[f"Q{j}" for j in range(1, 6)]).to_csv(
        os.path.join(OUT, "spatial_targeting", "figure7_transition_matrix_row_pct.csv"),
        index=True)
    N["transition_diagonal_pct"] = [round(float(comp[i, i]), 1) for i in range(5)]

    # population share of inventory per regime cell (for the paper text)
    N["pop_share"] = {k: round(float(v), 1) for k, v in gsum["pop_share_pct"].items()}
    N["regime_sizes"] = {k: int(v) for k, v in gsum["N"].items()}
    N["regime_stats"] = (gsum.round(4).replace({np.nan: None})
                         .reset_index().to_dict(orient="records"))
    N["labelled_per_cell"] = {k: int((cells_lab == k).sum()) for k in order_cells}

    from extras_v52 import step9_extras
    step9_extras(N, df300, df98, cells_lab, order_cells, Kstar, spatial_feats, CSV)

    # clean private keys
    for k in ["_y", "_g", "_oofs"]:
        N.pop(k, None)
    json.dump(N, open(os.path.join(HERE, "NUMBERS_V52.json"), "w"), indent=1, default=float)
    log("WROTE " + os.path.join(HERE, "NUMBERS_V52.json"))


if __name__ == "__main__":
    main()
