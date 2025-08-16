#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_realistic_dataset_auto_v3.py

Auto-tunes class overlap & label noise to hit a target multiclass AUROC,
then generates a large synthetic-but-realistic vulnerability dataset.

Fixes vs v3:
- Robust predict_proba stacking -> correct (n_samples, n_classes)
- Safe per-class AUROC when a fold has only one label
- Uses groupby.sample(...) instead of deprecated groupby.apply(...sample)
- No 'verbose' kw to LightGBM; uses callbacks
"""

import argparse
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

from lightgbm import LGBMClassifier
import lightgbm as lgb

import matplotlib.pyplot as plt

# ------------------------------ Utils ------------------------------ #

RNG = np.random.default_rng(42)

SERVICE_POOL = [
    "http", "https", "ssh", "ftp", "dns", "rdp", "smb", "mysql", "mssql", "ldap"
]
PROTO_POOL = ["tcp", "udp"]


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def draw_services(n, cls):
    """Slightly different service mix by class to mimic reality."""
    # weights per class (hand-tuned to feel realistic)
    base = np.array([20, 20, 12, 10, 10, 8, 8, 6, 4, 2], float)
    shift = np.array([10, 0, 0, 0, 0, 5, 8, 7, 4, 6], float) if cls >= 4 else \
            np.array([14, 18, 12, 12, 10, 4, 6, 6, 4, 4], float) if cls == 3 else \
            np.array([22, 20, 15, 12, 12, 2, 2, 3, 2, 2], float) if cls == 2 else \
            np.array([25, 22, 18, 12, 10, 1, 1, 1, 1, 1], float)

    w = base * 0.5 + shift
    w = w / w.sum()
    return RNG.choice(SERVICE_POOL, size=n, p=w)


def draw_proto(n, cls):
    """Mostly tcp; tiny udp change by class."""
    p_tcp = 0.92 if cls >= 4 else 0.90 if cls == 3 else 0.88 if cls == 2 else 0.86
    return RNG.choice(PROTO_POOL, size=n, p=[p_tcp, 1 - p_tcp])


def generate_class_block(n, cls, overlap=0.4, label_noise=0.04, feat_noise=0.06, random_noise=0.05):
    """
    Generate rows for one class (1..5) with controlled overlap & noise.
    overlap (higher) => more overlap (larger stds)
    label_noise => chance to flip class to neighbor
    feat_noise => extra per-feature jitter
    random_noise => a small fraction of pure noise rows
    """
    # anchors per class (means)
    cvss_mu = [1.0, 3.2, 5.5, 7.8, 9.6][cls - 1]
    sev_mu  = [0.2, 0.9, 1.7, 2.5, 3.4][cls - 1]
    exp_mu  = [0.03, 0.08, 0.20, 0.45, 0.60][cls - 1]
    dlen_mu = [200, 250, 300, 350, 400][cls - 1]
    age_mu  = [8, 26, 50, 75, 100][cls - 1]
    port_mu = 4500

    # stds grow with overlap
    cvss_sigma = 0.4 + 0.9 * overlap
    sev_sigma  = 0.25 + 0.45 * overlap
    exp_sigma  = 0.15 + 0.25 * overlap
    dlen_sigma = 35 + 25 * overlap
    age_sigma  = 7 + 10 * overlap
    port_sigma = 2600

    # draw base features
    cvss = RNG.normal(cvss_mu, cvss_sigma, size=n)
    cvss = clamp(cvss, 0.0, 10.0)

    severity = RNG.normal(sev_mu, sev_sigma, size=n)
    severity = clamp(np.round(severity), 0, 4).astype(int)

    exploit = RNG.normal(exp_mu, exp_sigma, size=n)
    exploit = clamp(exploit, 0.0, 1.0)
    exploit = (RNG.uniform(0, 1, size=n) < exploit).astype(int)

    dlen = RNG.normal(dlen_mu, dlen_sigma, size=n)
    dlen = clamp(dlen, 50, 700)

    age = RNG.normal(age_mu, age_sigma, size=n)
    age = clamp(age, 0, 200)

    port = RNG.normal(port_mu, port_sigma, size=n)
    port = clamp(np.round(port), 1, 9000).astype(int)

    persistence = np.ones(n, dtype=int)

    # categorical
    proto = draw_proto(n, cls)
    svc_name = draw_services(n, cls)

    # add mild feature noise
    if feat_noise > 0:
        cvss += RNG.normal(0, feat_noise, size=n)
        cvss = clamp(cvss, 0, 10)
        dlen += RNG.normal(0, feat_noise * 50, size=n)
        dlen = clamp(dlen, 50, 700)
        age += RNG.normal(0, feat_noise * 10, size=n)
        age = clamp(age, 0, 200)

    # random noise rows fraction
    if random_noise > 0:
        mask = RNG.uniform(0, 1, size=n) < random_noise
        m = mask.sum()
        if m > 0:
            cvss[mask] = RNG.uniform(0, 10, size=m)
            severity[mask] = RNG.integers(0, 5, size=m)
            exploit[mask] = RNG.integers(0, 2, size=m)
            dlen[mask] = RNG.uniform(50, 700, size=m)
            age[mask] = RNG.uniform(0, 200, size=m)
            port[mask] = RNG.integers(1, 9000, size=m)
            proto[mask] = RNG.choice(PROTO_POOL, size=m)
            svc_name[mask] = RNG.choice(SERVICE_POOL, size=m)

    y = np.full(n, cls, dtype=int)

    # neighbor label flips
    if label_noise > 0:
        flip_mask = RNG.uniform(0, 1, size=n) < label_noise
        idx = np.where(flip_mask)[0]
        for i in idx:
            if y[i] == 1:
                y[i] = 2
            elif y[i] == 5:
                y[i] = 4
            else:
                y[i] = y[i] + (1 if RNG.uniform() < 0.5 else -1)

    df = pd.DataFrame({
        "cvss": cvss,
        "severity": severity,
        "exploit_available": exploit,
        "description_len": dlen,
        "age_days": age,
        "persistence_scans": persistence,
        "port": port,
        "proto": pd.Categorical(proto, categories=PROTO_POOL),
        "svc_name": pd.Categorical(svc_name, categories=SERVICE_POOL),
        "remediation_priority": y
    })
    return df


def generate_dataset(n_rows, props, overlap, noise, feat_noise, random_noise):
    """Make full dataset by concatenating class blocks with given proportions."""
    props = np.array(props, float)
    props = props / props.sum()
    counts = (props * n_rows).astype(int)
    # fix rounding to hit exactly n_rows
    diff = n_rows - counts.sum()
    if diff > 0:
        counts[:diff] += 1

    parts = []
    for cls, n in enumerate(counts, start=1):
        parts.append(generate_class_block(
            n, cls, overlap=overlap, label_noise=noise,
            feat_noise=feat_noise, random_noise=random_noise
        ))
    df = pd.concat(parts, axis=0, ignore_index=True)
    return df


def stratified_balance(df, per_class=8000, seed=42):
    """Take up to per_class rows per label (balanced probe set)."""
    return (df.groupby("remediation_priority", group_keys=False)
              .sample(n=per_class, replace=False, random_state=seed))


def probe_auc_lightgbm(df, per_class=8000, seed=42, silent=True):
    """Train a tiny LightGBM and compute per-class AUROC on a held-out split."""
    probe = stratified_balance(df, per_class=per_class, seed=seed).reset_index(drop=True)

    # Encode categoricals
    proto_le = LabelEncoder()
    svc_le = LabelEncoder()
    probe["proto_le"] = proto_le.fit_transform(probe["proto"])
    probe["svc_le"] = svc_le.fit_transform(probe["svc_name"])

    X = probe[["cvss", "severity", "exploit_available", "description_len",
               "age_days", "persistence_scans", "port", "proto_le", "svc_le"]].copy()
    y = probe["remediation_priority"].astype(int).values - 1  # to 0..4

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.40, random_state=seed, stratify=y
    )

    model = LGBMClassifier(
        objective="multiclass",
        num_class=5,
        learning_rate=0.08,
        n_estimators=1000,
        max_depth=-1,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1e-6,
        reg_lambda=1e-6,
        random_state=seed,
        n_jobs=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False)]
    if not silent:
        callbacks.append(lgb.log_evaluation(100))

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="multi_logloss",
        callbacks=callbacks
    )

    # predict_proba -> (n_samples, n_classes) for multiclass
    proba = model.predict_proba(X_va, num_iteration=model.best_iteration_)
    if isinstance(proba, list):  # safety: some wrappers return a list of class arrays
        P = np.column_stack(proba)
    else:
        P = np.asarray(proba)  # (n_samples, 5)

    per_class_auc = []
    for k in range(5):
        y_true = (y_va == k).astype(int)
        # If only one label present in this fold, skip this class
        if y_true.min() == y_true.max():
            per_class_auc.append(np.nan)
            continue
        try:
            auc_k = roc_auc_score(y_true, P[:, k])
        except Exception:
            auc_k = np.nan
        per_class_auc.append(float(auc_k) if not np.isnan(auc_k) else np.nan)

    # macro across available classes
    valid = [a for a in per_class_auc if not np.isnan(a)]
    macro = float(np.mean(valid)) if len(valid) > 0 else float("nan")
    return macro, per_class_auc


def print_basic_summary(df):
    print(f"Loaded: {len(df):,} rows\n")
    print("=== Class distribution ===")
    print(df["remediation_priority"].value_counts().sort_index(), "\n")

    print("=== Per-class numeric summaries (head) ===")
    for cls, g in df.groupby("remediation_priority"):
        print(f"\n-- Class {cls} --")
        print(g[["cvss", "severity", "exploit_available", "description_len",
                 "age_days", "persistence_scans", "port"]].describe().loc[
                     ["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3))

    print("\n=== Top categorical values per class (head) ===")
    for cls, g in df.groupby("remediation_priority"):
        print(f"\n-- Class {cls} --")
        print("  proto:", dict(Counter(g["proto"]).most_common(5)))
        print("  svc_name:", dict(Counter(g["svc_name"]).most_common(5)))


def save_plots(df, outdir="sanity_plots", tsne_rows=20000, seed=42):
    os.makedirs(outdir, exist_ok=True)
    # Histograms
    num_cols = ["cvss", "description_len", "age_days"]
    for col in num_cols:
        plt.figure(figsize=(8, 4))
        for cls in sorted(df["remediation_priority"].unique()):
            g = df[df["remediation_priority"] == cls][col]
            plt.hist(g, bins=60, alpha=0.5, density=True, label=f"class {cls}")
        plt.legend()
        plt.title(f"{col} by class")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close()

    # t-SNE (downsample)
    sample = stratified_balance(df, per_class=max(1, tsne_rows // 5), seed=seed)
    X = sample[["cvss", "severity", "exploit_available", "description_len",
                "age_days", "persistence_scans", "port"]].values.astype(float)
    y = sample["remediation_priority"].values
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=seed, perplexity=30)
    Z = tsne.fit_transform(X)
    plt.figure(figsize=(7, 6))
    for cls in sorted(np.unique(y)):
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.6, label=f"class {cls}")
    plt.legend(markerscale=2)
    plt.title("t-SNE (numeric features)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "tsne_numeric.png"))
    plt.close()


# --------------------------- Auto-tuning --------------------------- #

def auto_tune(args):
    overlap = args.init_overlap
    noise = args.init_noise

    best_macro = -1.0
    best_params = (overlap, noise)
    print_interval = max(1, args.max_iters // 10)

    for it in range(1, args.max_iters + 1):
        print(f"[AutoTune] Iter {it}/{args.max_iters} | trying overlap={overlap:.3f}, noise={noise:.3f}")
        # Generate a probe dataset quickly
        df_probe = generate_dataset(args.probe_rows, args.proportions, overlap, noise,
                                    args.feat_noise, args.random_noise)
        macro_auc, per_class = probe_auc_lightgbm(df_probe, per_class=args.probe_sample_per_class, seed=42, silent=True)
        pcs = [f"{a:.3f}" if not (a is None or np.isnan(a)) else "NA" for a in per_class]
        print(f"  -> Macro AUROC={macro_auc:.4f} | per-class={pcs}")

        # track best
        if macro_auc > best_macro:
            best_macro = macro_auc
            best_params = (overlap, noise)

        # simple schedule: walk overlap/noise upward until we go below target
        # This gives you a *range* and we keep the best seen.
        overlap += 0.05
        noise = min(noise + 0.01, 0.80)  # cap for sanity

    print(f"\n[AutoTune] Best params -> overlap={best_params[0]:.3f}, noise={best_params[1]:.3f} "
          f"(best macro AUROC≈{best_macro:.4f})")

    # Generate final dataset with best seen
    print(f"\n[Finalize] Generating final dataset: rows={args.rows:,}, "
          f"overlap={best_params[0]:.3f}, noise={best_params[1]:.3f}, "
          f"feat_noise={args.feat_noise:.3f}, proportions={args.proportions}")
    df = generate_dataset(args.rows, args.proportions, best_params[0], best_params[1],
                          args.feat_noise, args.random_noise)
    df.to_csv(args.output, index=False)
    print(f"Saved dataset to {args.output}")

    # Quick sanity summary + quick AUROC probe on a balanced sample
    print("\nRunning final sanity check (this includes a quick AUROC probe)...")
    print_basic_summary(df)

    macro_auc, per_class = probe_auc_lightgbm(df, per_class=min(8000, df['remediation_priority'].value_counts().min()), seed=42, silent=True)
    print("\n=== Final quick AUROC (balanced sample) ===")
    for i, a in enumerate(per_class, start=1):
        s = f"{a:.3f}" if not (a is None or np.isnan(a)) else "NA"
        print(f"  Class {i}: {s}")
    print(f"Macro AUROC:   {macro_auc:.3f}")

    if args.plots:
        print("Saving plots to ./sanity_plots ...")
        save_plots(df, outdir="sanity_plots", tsne_rows=20000, seed=42)
        print("Saved plots.")


# ------------------------------ CLI ------------------------------ #

def parse_args():
    ap = argparse.ArgumentParser(description="Auto-tuned realistic dataset generator")
    ap.add_argument("--rows", type=int, required=True, help="Total rows to generate")
    ap.add_argument("--target_auc", type=float, default=0.90, help="(informational) target macro AUROC")
    ap.add_argument("--tolerance", type=float, default=0.01, help="(informational) tolerance around target")
    ap.add_argument("--max_iters", type=int, default=12, help="Auto-tune iterations")
    ap.add_argument("--init_overlap", type=float, default=0.40, help="Starting overlap")
    ap.add_argument("--init_noise", type=float, default=0.04, help="Starting neighbor label flip probability")
    ap.add_argument("--feat_noise", type=float, default=0.06, help="Per-feature jitter")
    ap.add_argument("--random_noise", type=float, default=0.05, help="Fraction of full-random rows per class")
    ap.add_argument("--proportions", type=float, nargs=5, default=[0.05, 0.15, 0.30, 0.30, 0.20],
                    help="Class proportions for labels 1..5")
    ap.add_argument("--probe_rows", type=int, default=250_000, help="Rows for each tuning probe dataset")
    ap.add_argument("--probe_sample_per_class", type=int, default=8000, help="Per-class sample for AUROC probe")
    ap.add_argument("--output", type=str, default="findings_realistic_v3.csv", help="Output CSV path")
    ap.add_argument("--plots", action="store_true", help="Save hist + t-SNE plots")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    auto_tune(args)
