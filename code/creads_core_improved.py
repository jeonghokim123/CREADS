#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CREADS 2023 — IMPROVED pipeline (surgical override of the original creads_core).

The original `creads_core.py` is imported UNMODIFIED. Only three of its
pipeline steps are replaced in-memory (monkeypatched), so every downstream
figure / CSV / targeting artefact keeps the original output contract and
file names, while the HDI model itself is upgraded:

  STEP 2  + country-median imputation and a 24-feature expanded set
  STEP 3  + hierarchical (country-mean-demeaned) ensemble target,
            manual 5-fold CV reporting overall R2 AND within-country R2
  STEP 4  + country-mean restoration when predicting the 584 regions

Additionally a circularity-free (spatial-only) GMM comparison CSV is written
so the regime classifier's dependence on predicted HDI can be documented.

Run:
    python creads_core_improved.py            # full pipeline
    python creads_core_improved.py --smoke    # steps 1-5 only, no figures
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# ── import the ORIGINAL, unmodified pipeline module ─────────────────────────
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent / 'creads_bundle' / 'python',
              Path('/content/ram_fpca/python')):
    if (_cand / 'creads_core.py').exists():
        sys.path.insert(0, str(_cand))
        break
import creads_core as cc  # noqa: E402

# capture the ORIGINAL callables before any monkeypatching
_ORIG_STEP2 = cc.step2_features
_ORIG_STEP5 = cc.step5_gmm

# ── expanded feature set (13 original -> 24) ────────────────────────────────
FEATS_IMPROVED = [
    'nightlight', 'log_nightlight',
    'nl_1km', 'nl_5km', 'nl_10km', 'nl_25km', 'nl_50km',
    'border_distance', 'log_pop_density', 'pop_sum',
    'urban_ratio', 'tree_cover', 'crop_ratio',
    # ── added ──
    'log_dist_capital', 'log_dist_airport',
    'grass', 'bare', 'water', 'shrub',
    'smod_urban', 'smod_rural', 'neighbors', 'area', 'nl_ratio',
]

# circularity-free GMM feature set (no HDI_predicted)
GMM_SPATIAL_FEATS = ['nightlight', 'border_distance', 'pop_density',
                     'log_dist_capital', 'log_dist_airport',
                     'urban_ratio', 'smod_urban']


# =============================================================================
# STEP 2 override — country-median imputation + expanded feature engineering
# =============================================================================
def step2_features_improved(df98, df300, verbose=True):
    df98, df300, avail_buf = _ORIG_STEP2(df98, df300, verbose=verbose)

    for df in (df98, df300):
        # ── country-median imputation for the two heavy-tailed covariates ──
        for f in ['pop_density', 'urban_ratio_25km']:
            if f in df.columns:
                if 'ISO3' in df.columns:
                    df[f] = df.groupby('ISO3')[f].transform(
                        lambda s: s.fillna(s.median()))
                df[f] = df[f].fillna(df[f].median())

        if 'pop_density' in df.columns:
            df['log_pop_density'] = np.log1p(df['pop_density'].clip(lower=0))

        # ── urban ratio: prefer the 25 km buffer definition ──
        if 'urban_ratio_25km' in df.columns:
            df['urban_ratio'] = df['urban_ratio_25km']

        # ── expanded covariates ──
        if 'dist_to_capital_km' in df.columns:
            df['log_dist_capital'] = np.log1p(df['dist_to_capital_km'])
        if 'dist_to_nearest_airport_km' in df.columns:
            df['log_dist_airport'] = np.log1p(df['dist_to_nearest_airport_km'])
        for src, dst in [('lc_grass_ratio', 'grass'), ('lc_bare_ratio', 'bare'),
                         ('lc_water_ratio', 'water'), ('lc_shrub_ratio', 'shrub'),
                         ('smod_urban_ratio', 'smod_urban'),
                         ('smod_rural_ratio', 'smod_rural'),
                         ('num_neighbors', 'neighbors')]:
            if src in df.columns:
                df[dst] = df[src]
        if 'area_km2' in df.columns:
            df['area'] = np.log1p(df['area_km2'])
        if 'viirs_mean_1km' in df.columns and 'viirs_mean_50km' in df.columns:
            df['nl_ratio'] = df['viirs_mean_1km'] / (df['viirs_mean_50km'] + 1e-6)

        # median-fill any residual NaNs in engineered covariates
        for f in FEATS_IMPROVED:
            if f in df.columns and df[f].notna().any():
                df[f] = df[f].fillna(df[f].median())

    if verbose:
        print(f"  [improved] expanded feature engineering -> "
              f"{len([f for f in FEATS_IMPROVED if f in df98.columns])} features")

    return df98, df300, avail_buf


# =============================================================================
# shared ensemble helpers
# =============================================================================
def _fit_ensemble(X_raw, y, seed=2024):
    imputer = SimpleImputer(strategy='median').fit(X_raw)
    X = imputer.transform(X_raw)
    scaler = StandardScaler().fit(X)
    X_sc = scaler.transform(X)

    rf = RandomForestRegressor(n_estimators=300, max_depth=12,
                               min_samples_leaf=2, n_jobs=-1, random_state=seed)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=seed)
    ridge = Ridge(alpha=1.0)

    rf.fit(X, y)
    gb.fit(X, y)
    ridge.fit(X_sc, y)

    w = np.clip([r2_score(y, rf.predict(X)),
                 r2_score(y, gb.predict(X)),
                 r2_score(y, ridge.predict(X_sc))], 0, None)
    if w.sum() <= 0:
        w = np.array([1.0, 1.0, 1.0])
    w = w / w.sum()
    return dict(imputer=imputer, scaler=scaler, rf=rf, gb=gb, ridge=ridge, weights=w)


def _predict_ensemble(M, X_raw):
    X = M['imputer'].transform(X_raw)
    X_sc = M['scaler'].transform(X)
    w = M['weights']
    return (w[0] * M['rf'].predict(X) +
            w[1] * M['gb'].predict(X) +
            w[2] * M['ridge'].predict(X_sc))


def _country_means(df, valid_mask):
    sub = df.loc[valid_mask]
    cm = sub.groupby('ISO3')['HDI'].mean()
    return cm, float(sub['HDI'].mean())


def _within_r2(y, pred, groups):
    ym = pd.Series(np.asarray(y)).groupby(np.asarray(groups)).transform('mean').values
    pm = pd.Series(np.asarray(pred)).groupby(np.asarray(groups)).transform('mean').values
    return r2_score(np.asarray(y) - ym, np.asarray(pred) - pm)


# =============================================================================
# STEP 3 override — hierarchical (demeaned) ensemble + honest CV
# =============================================================================
def step3_train_model_improved(df98, verbose=True):
    if verbose:
        print("\n" + "=" * 72)
        print("  STEP 3 | HDI ensemble model training  [IMPROVED: hierarchical]")
        print("=" * 72)

    avail = [f for f in FEATS_IMPROVED if f in df98.columns]
    if verbose:
        print(f"  Features used ({len(avail)}): {avail}")

    valid_mask = df98['HDI'].notna().values
    X_raw = df98.loc[valid_mask, avail].values.astype(float)
    y = df98.loc[valid_mask, 'HDI'].values.astype(float)
    groups = df98.loc[valid_mask, 'ISO3'].values

    # ── hierarchical target: HDI minus its country mean ──
    cm, gmu = _country_means(df98, valid_mask)
    y_dm = y - pd.Series(groups).map(cm).fillna(gmu).values

    M = _fit_ensemble(X_raw, y_dm)
    y_en_dm = _predict_ensemble(M, X_raw)
    y_en = y_en_dm + pd.Series(groups).map(cm).fillna(gmu).values

    ens_r2 = r2_score(y, y_en)
    ens_rmse = np.sqrt(((y - y_en) ** 2).mean())
    ens_r2_within = _within_r2(y, y_en, groups)

    # ── honest 5-fold CV (manual, so country means are fold-safe) ──
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    fold_r2 = []
    for tr, te in kf.split(X_raw):
        cm_tr = pd.Series(groups[tr]).to_frame('ISO3').assign(
            HDI=y[tr]).groupby('ISO3')['HDI'].mean()
        gmu_tr = float(y[tr].mean())
        y_tr_dm = y[tr] - pd.Series(groups[tr]).map(cm_tr).fillna(gmu_tr).values
        Mf = _fit_ensemble(X_raw[tr], y_tr_dm)
        p = _predict_ensemble(Mf, X_raw[te]) + \
            pd.Series(groups[te]).map(cm_tr).fillna(gmu_tr).values
        oof[te] = p
        fold_r2.append(r2_score(y[te], p))

    cv_mean = float(np.mean(fold_r2))
    cv_std = float(np.std(fold_r2))
    cv_overall = float(r2_score(y, oof))
    cv_within = float(_within_r2(y, oof, groups))

    if verbose:
        print(f"\n  ── In-sample performance (hierarchical) ──")
        print(f"    Ensemble          R² = {ens_r2:.4f}  RMSE = {ens_rmse:.4f}")
        print(f"    Within-country R² = {ens_r2_within:.4f}")
        print(f"    Weights: RF={M['weights'][0]:.3f}, GB={M['weights'][1]:.3f}, "
              f"Ridge={M['weights'][2]:.3f}")
        print(f"\n  ── 5-Fold Cross-Validation (hierarchical, fold-safe) ──")
        print(f"    Overall CV R² (per-fold mean) = {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"    Overall CV R² (pooled OOF)    = {cv_overall:.4f}")
        print(f"    Within-country CV R²          = {cv_within:.4f}")

    out = dict(M)
    out.update(dict(features=avail, r2=ens_r2, rmse=ens_rmse,
                    r2_within=ens_r2_within,
                    cv_mean=cv_mean, cv_std=cv_std,
                    cv_overall=cv_overall, cv_within=cv_within,
                    country_means=cm, global_mean=gmu,
                    improved=True))
    return out


# =============================================================================
# STEP 4 override — restore country means on the 584-region prediction
# =============================================================================
def step4_predict_improved(df300, model, verbose=True):
    if verbose:
        print("\n" + "=" * 72)
        print("  STEP 4 | Predict HDI for 300 regions  [IMPROVED: +country mean]")
        print("=" * 72)

    feats = model['features']
    X_raw = np.zeros((len(df300), len(feats)))
    for i, f in enumerate(feats):
        if f in df300.columns:
            col = df300[f]
            fill = col.median() if col.notna().any() else 0.0
            X_raw[:, i] = col.fillna(fill).values
        else:
            X_raw[:, i] = 0.0

    base = _predict_ensemble(model, X_raw)
    if 'ISO3' in df300.columns:
        add = df300['ISO3'].map(model['country_means']).fillna(model['global_mean']).values
    else:
        add = np.full(len(df300), model['global_mean'])
    pred = np.clip(base + add, 0.30, 0.99)
    df300['HDI_predicted'] = pred

    if verbose:
        print(f"  ✓ mean={pred.mean():.4f} ± {pred.std():.4f}  "
              f"range=[{pred.min():.4f}, {pred.max():.4f}]")
    return df300


# =============================================================================
# STEP 5 wrapper — keep original GMM, add circularity-free comparison CSV
# =============================================================================
def step5_gmm_wrapper(df300, K=3, verbose=True):
    df300, gmm_meta = _ORIG_STEP5(df300, K=K, verbose=verbose)

    # ── circularity-free (spatial-only) GMM sensitivity artefact ──
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score
        feats = [f for f in GMM_SPATIAL_FEATS if f in df300.columns]
        X = SimpleImputer(strategy='median').fit_transform(
            df300[feats].values.astype(float))
        X_sc = StandardScaler().fit_transform(X)
        gm = GaussianMixture(n_components=K, covariance_type='full',
                             random_state=2024, n_init=20).fit(X_sc)
        lab = gm.predict(X_sc)
        sil = silhouette_score(X_sc, lab) if len(np.unique(lab)) > 1 else np.nan
        os.makedirs(cc.CSV_DIR, exist_ok=True)
        comp = pd.DataFrame({
            'specification': ['original (with HDI_predicted)', 'circularity-free (spatial only)'],
            'features': ['nightlight,border_distance,pop_density,HDI_predicted',
                         ','.join(feats)],
            'BIC': [gmm_meta['bic'], float(gm.bic(X_sc))],
            'AIC': [gmm_meta['aic'], float(gm.aic(X_sc))],
            'silhouette': [gmm_meta['silhouette'], float(sil)],
        })
        comp.to_csv(f'{cc.CSV_DIR}/gmm_circularity_free_comparison.csv', index=False)
        if verbose:
            print(f"  [improved] circularity-free GMM silhouette = {sil:.4f} "
                  f"(comparison CSV written)")
    except Exception as e:  # never let the sensitivity artefact break the run
        print(f"  [improved] circularity-free GMM skipped: {e}")

    return df300, gmm_meta


# =============================================================================
# entry point
# =============================================================================
def _patch():
    cc.step2_features = step2_features_improved
    cc.step3_train_model = step3_train_model_improved
    cc.step4_predict = step4_predict_improved
    cc.step5_gmm = step5_gmm_wrapper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true',
                    help='run steps 1-5 only (no figures) and print metrics')
    args = ap.parse_args()
    _patch()

    if args.smoke:
        df98, df300 = cc.step1_load()
        df98, df300, avail_buf = cc.step2_features(df98, df300)
        model = cc.step3_train_model(df98)
        df300 = cc.step4_predict(df300, model)
        df300, gmm_meta = cc.step5_gmm(df300)
        print("\nSMOKE OK | cv_overall=%.4f cv_within=%.4f "
              "pred_mean=%.4f gmm_sil=%.4f"
              % (model['cv_overall'], model['cv_within'],
                 df300['HDI_predicted'].mean(), gmm_meta['silhouette']))
        return

    cc.run()


if __name__ == '__main__':
    main()
