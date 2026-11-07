#!/usr/bin/env python3
"""Two-level extension for Revised_Manuscript(3): luminosity-free country
clustering (Step 2) + within-cluster re-estimation of the SAME outcome-
independent 3-feature accessibility regimes (Step 3). No welfare outcome and
no nighttime light enters the regime definition at any level."""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import creads_core as cc, creads_core_improved as ci
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

SEED = 2024
BASE = os.path.dirname(HERE); OUT = os.path.join(BASE, "outputs_ext"); CSV = os.path.join(OUT, "csv_exports")
os.makedirs(CSV, exist_ok=True)
cc.BASE = BASE; cc.DATA_DIR = os.path.join(BASE, "data")
cc.FILE_98 = os.path.join(cc.DATA_DIR, "MATCHED_340REGIONS_WITH_GDL_2023.xlsx")
cc.FILE_300 = os.path.join(cc.DATA_DIR, "COMPLETE_584REGIONS_WITH_CONNECTIVITY_2023.xlsx")
df98, df300 = cc.step1_load(verbose=False)
df98, df300, _ = ci.step2_features_improved(df98, df300, verbose=False)

# ---------- Step 2: country clustering, luminosity-free ----------
g = df300.groupby("ISO3")
C = pd.DataFrame({
    "popd": np.log1p(g["pop_density"].mean().clip(lower=0)),
    "urban": g["urban_ratio"].mean(),
    "border": np.log1p(g["border_distance"].mean().clip(lower=0)),
    "cap": np.log1p(g["dist_to_capital_km"].mean().clip(lower=0)),
    "air": np.log1p(g["dist_to_nearest_airport_km"].mean().clip(lower=0)),
    "smod": g["smod_urban_ratio"].mean(),
    "tree": g["tree_cover"].mean()})
CF = ["popd", "urban", "border", "cap", "air", "smod", "tree"]
X = C[CF].values.astype(float)
Xsc = (X - X.mean(0)) / (X.std(0) + 1e-9)
rows, fits = [], {}
for K in range(2, 7):
    gm = GaussianMixture(n_components=K, covariance_type="full", random_state=SEED,
                         n_init=20).fit(Xsc)
    lab = gm.predict(Xsc)
    rows.append(dict(K=K, BIC=gm.bic(Xsc), AIC=gm.aic(Xsc),
                     silhouette=float(silhouette_score(Xsc, lab)),
                     min_countries=int(pd.Series(lab).value_counts().min())))
    fits[K] = (gm, lab)
sweep = pd.DataFrame(rows); sweep.to_csv(os.path.join(CSV, "ext_country_sweep.csv"), index=False)
bic_argmin = int(sweep.loc[sweep.BIC.idxmin(), "K"])
Kstar = next((int(r.K) for r in sweep.sort_values("BIC").itertuples() if r.min_countries >= 3), bic_argmin)
guard = Kstar != bic_argmin
gm, raw = fits[Kstar]
C["cluster"] = 0  # set below
order = pd.Series(C["popd"].values).groupby(raw).mean().sort_values().index.tolist()
remap = {o: i + 1 for i, o in enumerate(order)}
C["cluster"] = [remap[r] for r in raw]
MEM = C.reset_index().rename(columns={"index": "ISO3"})
MEM.to_csv(os.path.join(CSV, "ext_country_membership.csv"), index=False)

# ---------- Step 3: within-cluster regional regimes (same 3 features) ----------
df300 = df300.merge(MEM[["ISO3", "cluster"]], on="ISO3", how="left")
FE3 = ["border_distance", "log_pop_density", "log_dist_capital"]
rs_rows, cells_meta = [], []
for c in range(1, Kstar + 1):
    sub = df300[df300.cluster == c]
    Xr = SimpleImputer3(sub[FE3]) if False else __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median").fit_transform(sub[FE3].values.astype(float))
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    Z = (Xr - mu) / sd
    cands = []
    for K in (2, 3):
        gmk = GaussianMixture(n_components=K, covariance_type="full", random_state=SEED,
                              n_init=20).fit(Z)
        labk = gmk.predict(Z)
        mincell = int(pd.Series(labk).value_counts().min())
        rs_rows.append(dict(cluster=c, K=K, BIC=gmk.bic(Z), AIC=gmk.aic(Z),
                            silhouette=float(silhouette_score(Z, labk)), n=len(sub), min_cell=mincell))
        cands.append((gmk.bic(Z), K, labk, gmk, float(silhouette_score(Z, labk)), mincell))
    feas = [t for t in cands if t[5] >= 8]
    bic, Kc, labk, gmk, sil, mincell = sorted(feas or cands, key=lambda t: t[0])[0]
    names = {2: ["Sparse-Interior", "Dense-Core"],
             3: ["Sparse-Interior", "Transitional-Access", "Dense-Core"]}[Kc]
    ordt = pd.Series(sub["pop_density"].values).groupby(labk).mean().sort_values().index.tolist()
    rm = {o: i for i, o in enumerate(ordt)}
    typ = np.array([rm[l] for l in labk])
    df300.loc[sub.index, "regional_type"] = typ
    df300.loc[sub.index, "regime_label"] = [f"C{c}-{names[t]}" for t in typ]
    rs_rows.append(dict(cluster=c, selected_K=Kc))
    cells_meta.append((c, names, Kc, sil, mincell))
pd.DataFrame(rs_rows).to_csv(os.path.join(CSV, "ext_regional_sweep.csv"), index=False)
cells = sorted(df300.regime_label.unique())

# ---------- summary ----------
G = df300.groupby("regime_label")
S = pd.DataFrame({"N": G.size(), "viirs": G["viirs_mean"].mean(),
                  "popd": G["pop_density"].mean(), "pop": G["pop_sum"].sum()})
S["share_pct"] = S["N"] / S["N"].sum() * 100
S["pop_share_pct"] = S["pop"] / S["pop"].sum() * 100
j = df300[["rid", "regime_label"]].merge(df98.loc[df98["HDI"].notna(), ["rid", "HDI"]], on="rid", how="inner")
S["hdi_obs"] = j.groupby("regime_label")["HDI"].mean()
labn = j.groupby("regime_label").size()
S["n_labelled"] = labn.reindex(S.index).fillna(0).astype(int)
corr_nl = {}
for k, sub in j.groupby("regime_label"):
    s = sub.merge(df98[["rid", "viirs_mean"]], on="rid")
    if len(s) >= 8:
        corr_nl[k] = round(float(np.corrcoef(np.log1p(s.viirs_mean), s.HDI)[0, 1]), 3)
S = S.reindex(cells)
S.round(4).to_csv(os.path.join(CSV, "ext_cell_summary.csv"))

# ---------- diagnostics ----------
feats3 = [f for f in FE3 if f in df300.columns]
Xi = __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median").fit_transform(df300[feats3].values.astype(float))
Zp = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-9)
sil_flat_cells = float(silhouette_score(Zp, df300.regime_label.values))
nmi_cells = float(normalized_mutual_info_score(df300.ISO3.astype(str), df300.regime_label))
rng = np.random.default_rng(11)
aris_c = []
for b in range(200):
    idxb = rng.integers(0, len(Xsc), len(Xsc))
    gmB = GaussianMixture(n_components=Kstar, covariance_type="full", random_state=SEED, n_init=10).fit(Xsc[idxb])
    uniq = np.unique(idxb); lb = gmB.predict(Xsc[uniq]); lp = np.array([remap[fits[Kstar][1][i]] for i in uniq])
    cont = np.zeros((Kstar, Kstar))
    for a in range(Kstar):
        for b2 in range(Kstar):
            cont[a, b2] = np.sum((lb == a) & (lp == b2))
    ra, cb = linear_sum_assignment(-cont)
    mp = {int(a): int(b2) for a, b2 in zip(ra, cb)}
    aris_c.append(adjusted_rand_score(lp, np.array([mp.get(int(x), -1) for x in lb])))
aris_r = []
for c in range(1, Kstar + 1):
    sub = df300[df300.cluster == c]
    Xr = __import__("sklearn.impute", fromlist=["SimpleImputer"]).SimpleImputer(strategy="median").fit_transform(sub[feats3].values.astype(float))
    Zs = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
    Kc = int(sub.regional_type.max()) + 1
    gmR = GaussianMixture(n_components=Kc, covariance_type="full", random_state=SEED, n_init=20).fit(Zs)
    pr = gmR.predict(Zs)
    for b in range(200):
        idxb = rng.integers(0, len(Zs), len(Zs))
        gmB = GaussianMixture(n_components=Kc, covariance_type="full", random_state=SEED, n_init=10).fit(Zs[idxb])
        uniq = np.unique(idxb); lb = gmB.predict(Zs[uniq]); lp = pr[uniq]
        cont = np.zeros((Kc, Kc))
        for a in range(Kc):
            for b2 in range(Kc):
                cont[a, b2] = np.sum((lb == a) & (lp == b2))
        ra, cb = linear_sum_assignment(-cont)
        mp = {int(a): int(b2) for a, b2 in zip(ra, cb)}
        aris_r.append(adjusted_rand_score(lp, np.array([mp.get(int(x), -1) for x in lb])))

# ---------- FPCA per cell (labelled) ----------
pc = [f"viirs_mean_{k}km" for k in [1, 5, 10, 25, 50]]
rawP = np.nan_to_num(df98.loc[df98["HDI"].notna(), pc].values.astype(float), nan=0.0)
LP = np.log1p(np.clip(rawP, 0, None)); SH = LP - LP.mean(1, keepdims=True)
cells_v = df98.loc[df98["HDI"].notna(), "rid"].map(dict(zip(df300.rid, df300.regime_label))).fillna("").values
fp = {}
for cell in cells:
    m = cells_v == cell
    if m.sum() < 12:
        continue
    P = SH[m]; mu, sd = P.mean(0), P.std(0) + 1e-9
    V = np.linalg.svd((P - mu) / sd, full_matrices=False)[2]
    ev = np.linalg.svd((P - mu) / sd, full_matrices=False)[1] ** 2; ev = ev / ev.sum()
    r2 = np.random.default_rng(23); cos = []
    for b in range(200):
        ib = r2.integers(0, len(P), len(P))
        Vb = np.linalg.svd((P[ib] - P[ib].mean(0)) / (P[ib].std(0) + 1e-9), full_matrices=False)[2][0]
        cos.append(abs(float(V[0] @ Vb)))
    fp[cell] = dict(n=int(m.sum()), pc1=round(float(ev[0] * 100), 1), cum2=round(float(ev[:2].sum() * 100), 1),
                    cos_med=round(float(np.median(cos)), 3))

# cross-cluster observed-HDI contrast for the same type
tt = S.copy(); tt["type"] = [k.split("-", 1)[1] for k in tt.index]
spread = {}
for t, grp in tt.groupby("type"):
    vals = grp["hdi_obs"].dropna()
    if len(vals) >= 2:
        spread[t] = dict(min_cluster=str(vals.idxmin()), max_cluster=str(vals.idxmax()),
                         spread=round(float(vals.max() - vals.min()), 3))
N = dict(K_star=Kstar, bic_argmin=bic_argmin, guard=bool(guard),
         sweep=sweep.round(4).to_dict("records"),
         clusters={str(c): dict(countries=int((MEM.cluster == c).sum()),
                                regions=int((df300.cluster == c).sum()),
                                members=sorted(MEM[MEM.cluster == c].ISO3),
                                inv_share=round(float((df300.cluster == c).mean() * 100), 1))
                   for c in range(1, Kstar + 1)},
         regional=dict((str(c), dict(K=nm[2], silhouette=round(nm[3], 3))) for c, nm in zip([x[0] for x in cells_meta], cells_meta)),
         n_cells=int(len(cells)), cells=cells,
         cell_summary=S.round(4).reset_index().to_dict("records"),
         sil_flat_cells=round(sil_flat_cells, 4), nmi_cells_country=round(nmi_cells, 4),
         ari_country=dict(median=round(float(np.median(aris_c)), 3), p05=round(float(np.percentile(aris_c, 5)), 3)),
         ari_regional=dict(median=round(float(np.median(aris_r)), 3), p05=round(float(np.percentile(aris_r, 5)), 3)),
         fpca_cells=fp, corr_nl=corr_nl, spread=spread,
         sil_by_cluster=dict((str(c), round(cm[3], 3)) for c, cm in zip([x[0] for x in cells_meta], cells_meta)))
json.dump(N, open(os.path.join(HERE, "NUMBERS_EXT.json"), "w"), indent=1, default=float)
print(json.dumps(N, indent=1, default=str)[:4600])

