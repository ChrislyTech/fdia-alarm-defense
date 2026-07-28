# -*- coding: utf-8 -*-
"""
main.py — DAE-PCA + VACF-style chattering FDIA en régime fault-free

Projet : FDIA sur détecteur DAE-PCA (TEP)
- PHASE 1  : Entraînement sur données normales (IDV=1)
- PHASE 2  : Calcul des seuils T² et SPE (99e percentile)
- PHASE 3A : Baseline FAR/FDR sur toutes les fautes IDV(1..21)
- PHASE 3B : Construction de SPE(x) pour le harness
- PHASE 3C : Expérience chattering FDIA fault-free (VACF)
"""

import os
import math
from typing import Dict, Any
import csv
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch as t
from torch import nn
from torch.autograd import Variable as V
import torch.optim as optim

import re
import config as opt
from dataset import dataset
from model import AE
from utils import BICcom, FDR, FAR
from exp_harness import ChatteringConfig, run_vacf_fault_free_experiment


# ============================================================
#  Helpers génériques
# ============================================================

def _to_tensor(x):
    """Convertit proprement en Tensor float32 (CPU)."""
    if isinstance(x, t.Tensor):
        return x.clone().detach().float()
    x_np = np.asarray(x, dtype=np.float32)
    return t.tensor(x_np, dtype=t.float32)


# ============================================================
#  DAE–PCA : entraînement / validation / seuils
# ============================================================

def loss_function(x, xp, h, hp):
    """Perte = λ1 * ||x - xp||^2  +  λ2 * ||h - hp||^2."""
    criterion = nn.MSELoss(reduction="mean")
    loss1 = criterion(xp, x)
    temp = h.detach()  # pas de rétroprop via h dans le 2e terme
    loss2 = criterion(hp, temp)
    loss = opt.la1 * loss1 + opt.la2 * loss2
    return loss, loss1, loss2


def train(ae: AE, train_x: t.Tensor, epoch: int):
    """Un epoch d’entraînement sur l’ensemble normal (IDV=1)."""
    ae.train()
    lr0 = opt.lr * math.pow(0.7, epoch // 350)
    optimizer = optim.Adam(ae.parameters(), lr=lr0)

    x = V(train_x, requires_grad=False)
    xp, T, h, hp, W = ae(x, opt)

    train_loss, l1, l2 = loss_function(x, xp, h, hp)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    return ae, train_loss.item(), l1.item(), l2.item()


def val(ae: AE, val_x) -> float:
    """Perte de validation sur données normales."""
    x_val = _to_tensor(val_x)
    ae.eval()
    with t.no_grad():
        x = V(x_val, requires_grad=False)
        xp, T, h, hp, W = ae(x, opt)
        val_loss, _, _ = loss_function(x, xp, h, hp)
    ae.train()
    return float(val_loss.item())


def thre(ae: AE, best_path: str, train_x):
    """
    Statistiques T² / SPE sur les données d’entraînement normales.
    Retourne : T2c, SPEc, cov_T, T_train
    """
    train_x_t = _to_tensor(train_x)

    ae.load_state_dict(t.load(best_path, map_location=t.device("cpu")))
    ae.eval()
    with t.no_grad():
        x = V(train_x_t)
        xp, T, h, hp, W = ae(x, opt)

    N = x.size(0)
    cov_T = t.inverse(T.t().mm(T) / (N - 1))

    T2c = []
    for i in range(N):
        ttc = T[i].unsqueeze(0)
        t2c = ttc.mm(cov_T).mm(ttc.t()).item()
        T2c.append(t2c)

    SPEc = t.sum((xp - x) ** 2, dim=1).squeeze().detach().cpu().numpy()
    T2c = np.array(T2c, dtype=np.float64)
    return T2c, SPEc, cov_T, T


def tes(best_path: str, test_x, cov_T: t.Tensor, J_T2: float, J_SPE: float):
    """
    Exécute le détecteur sur une trajectoire de test (potentiellement fautive).
    Retourne T², SPE et BIC par échantillon.
    """
    test_x_t = _to_tensor(test_x)

    ae = AE(opt)
    ae.load_state_dict(t.load(best_path, map_location=t.device("cpu")))
    ae.eval()

    T2, SPE, BIC = [], [], []

    with t.no_grad():
        x_all = V(test_x_t)

    num = x_all.size(0)
    for i in range(num):
        x = x_all[i, :].unsqueeze(0)
        xp, T, h, hp, _ = ae(x, opt)

        t2_val = T.mm(cov_T).mm(T.t()).item()
        spe_val = t.sum((xp - x) ** 2, dim=1).item()

        Pz_xF, Pz_Fx = BICcom(t2_val, J_T2, 0.99)
        Px_xF, Px_Fx = BICcom(spe_val, J_SPE, 0.99)
        bic_val = (Pz_xF * Pz_Fx + Px_xF * Px_Fx) / (Pz_xF + Px_xF + 1e-12)

        T2.append(t2_val)
        SPE.append(spe_val)
        BIC.append(bic_val)

    return T2, SPE, BIC


def eval_far_fdr(T2, SPE, BIC, f: int, J_T2: float, J_SPE: float, bic_thr: float = 0.01) -> Dict[str, float]:
    """Calcule FAR/FDR pour T², SPE et BIC sur la fenêtre [0,f)."""
    return dict(
        FDR_T2=float(FDR(T2, J_T2, f)),
        FAR_T2=float(FAR(T2, J_T2, f)),
        FDR_SPE=float(FDR(SPE, J_SPE, f)),
        FAR_SPE=float(FAR(SPE, J_SPE, f)),
        FDR_BIC=float(FDR(BIC, bic_thr, f)),
        FAR_BIC=float(FAR(BIC, bic_thr, f)),
    )


def calculate_threshold(values, alpha: float = 0.99) -> float:
    """Seuil = quantile alpha (e.g. 0.99 => 99e percentile)."""
    values = np.asarray(values, dtype=float)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return float(np.quantile(values, alpha_clamped))


# ============================================================
#  SPE et étape de perturbation pour le harness VACF
# ============================================================

def build_spe_fn(best_path: str):
    """Construit une fonction SPE(x) utilisable côté harness (numpy in/out)."""
    ae = AE(opt)
    ae.load_state_dict(t.load(best_path, map_location=t.device("cpu")))
    ae.eval()

    def spe_fn(x_np: np.ndarray):
        x = t.tensor(x_np, dtype=t.float32)
        with t.no_grad():
            if x.ndim == 1:
                x = x.unsqueeze(0)
                xp, _, _, _, _ = ae(x, opt)
                spe_vec = t.sum((xp - x) ** 2, dim=1)
                return float(spe_vec.item())
            elif x.ndim == 2:
                xp, _, _, _, _ = ae(x, opt)
                spe_vec = t.sum((xp - x) ** 2, dim=1)
                return spe_vec.detach().cpu().numpy()
            else:
                raise ValueError("spe_fn attend un array 1D ou 2D")

    return spe_fn


def plot_vacf_timeseries(out_dir: Path, J_SPE: float) -> None:
    """
    Plot full SPE trajectories (baseline vs attack) + alarms for VACF runs.

    This assumes that out_dir / "series.npz" contains at least:
        SPE_base, SPE_attack, alarms_base, alarms_attack
    and optionally: gate.
    """
    out_dir = Path(out_dir)
    series_path = out_dir / "series.npz"
    if not series_path.exists():
        print(f"[plot_vacf_timeseries] No series.npz in {out_dir}, skipping.")
        return

    data = np.load(series_path)

    spe_base = data.get("SPE_base")
    spe_attack = data.get("SPE_attack")
    alarms_base = data.get("alarms_base")
    alarms_attack = data.get("alarms_attack")
    gate = data.get("gate")

    if spe_base is None or spe_attack is None:
        print("[plot_vacf_timeseries] SPE_base or SPE_attack missing, skipping.")
        return

    N = len(spe_base)
    t = np.arange(N)

    # Small vertical scale for alarms so they sit near the bottom of the axes
    alarm_height = 0.2 * float(J_SPE)

    fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)

    # ------------------------------------------------------------------
    # Baseline (no attack)
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(t, spe_base, label="SPE (baseline)")
    ax1.axhline(J_SPE, linestyle="--", label=r"Threshold $J_{\mathrm{SPE}}$")
    if alarms_base is not None:
        ax1.step(t, alarms_base * alarm_height, where="post",
                 label="Alarm (baseline)")
    ax1.set_ylabel("SPE")
    ax1.set_title("Baseline (no attack)")
    ax1.legend(loc="upper right")

    # ------------------------------------------------------------------
    # Under chattering FDIA (fault-free)
    # ------------------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(t, spe_attack, label="SPE (attack)")
    ax2.axhline(J_SPE, linestyle="--", label=r"Threshold $J_{\mathrm{SPE}}$")
    if alarms_attack is not None:
        ax2.step(t, alarms_attack * alarm_height, where="post",
                 label="Alarm (attack)")
    if gate is not None:
        ax2.step(t, gate * alarm_height, where="post",
                 label="Gate ON")
    ax2.set_ylabel("SPE")
    ax2.set_xlabel("Sample index")
    ax2.set_title("Under chattering FDIA (fault-free)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    out_path = out_dir / "spe_alarms_base_vs_attack.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"📈 Full SPE+alarms figure saved to: {out_path}")


def plot_vacf_timeseries_zoom(
        out_dir: Path,
        J_SPE: float,
        t_start: int = 400,
        t_end: int = 800,
        y_max_cap: float | None = 1.0,
) -> None:
    """
    Zoomed view around the threshold for SPE(base) and SPE(attack),
    plus a separate alarm-vs-time figure.

    Figure 1 (spe_zoom_*):
        - 2 stacked subplots: SPE (baseline) and SPE (attack)
        - Same time window [t_start, t_end)
        - Red dots mark alarmed samples
        - Y-axis is zoomed to better show oscillations around J_SPE

    Figure 2 (alarms_zoom_*):
        - 1 subplot: alarm (attack) as a binary 0/1 bar over time.
    """
    out_dir = Path(out_dir)
    series_path = out_dir / "series.npz"
    if not series_path.exists():
        print(f"[plot_vacf_timeseries_zoom] No series.npz in {out_dir}, skipping.")
        return

    data = np.load(series_path)

    spe_base = data.get("SPE_base")
    spe_attack = data.get("SPE_attack")
    alarms_base = data.get("alarms_base")
    alarms_attack = data.get("alarms_attack")

    if spe_base is None or spe_attack is None:
        print("[plot_vacf_timeseries_zoom] SPE_base or SPE_attack missing, skipping.")
        return

    N = len(spe_base)
    t_start = max(0, int(t_start))
    t_end = min(N, int(t_end))
    if t_start >= t_end:
        print(f"[plot_vacf_timeseries_zoom] Invalid window [{t_start}, {t_end}), skipping.")
        return

    idx = np.arange(t_start, t_end)
    sb = spe_base[t_start:t_end]
    sa = spe_attack[t_start:t_end]
    ab = alarms_base[t_start:t_end] if alarms_base is not None else None
    aa = alarms_attack[t_start:t_end] if alarms_attack is not None else None

    # --------------------- vertical zoom around J_SPE -------------------
    ymax = float(max(np.max(sb), np.max(sa), J_SPE))
    if y_max_cap is not None:
        ymax = min(ymax, float(y_max_cap))
    ymax *= 1.05  # small headroom

    # Short label like "far10", "far20" for file names
    m = re.search(r"far(\d+)", out_dir.name)
    far_label = f"far{m.group(1)}" if m else "vacf"

    # ------------------------------------------------------------------
    # Figure 1: SPE (baseline & attack) only
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)

    # Baseline SPE
    ax1.plot(idx, sb, label="SPE (baseline)")
    ax1.axhline(J_SPE, linestyle="--", label=r"Threshold $J_{\mathrm{SPE}}$")
    if ab is not None:
        alarm_idx = idx[ab > 0.5]
        ax1.scatter(
            alarm_idx,
            sb[ab > 0.5],
            s=15,
            color="red",
            label="Alarm (baseline)",
        )
    ax1.set_ylabel("SPE (base)")
    ax1.set_ylim(0.0, ymax)
    ax1.legend(loc="upper right")

    # Attack SPE
    ax2.plot(idx, sa, label="SPE (attack)")
    ax2.axhline(J_SPE, linestyle="--", label=r"Threshold $J_{\mathrm{SPE}}$")
    if aa is not None:
        alarm_idx = idx[aa > 0.5]
        ax2.scatter(
            alarm_idx,
            sa[aa > 0.5],
            s=15,
            color="red",
            label="Alarm (attack)"
        )
    ax2.set_ylabel("SPE (attack)")
    ax2.set_xlabel("Time (samples)")
    ax2.set_ylim(0.0, ymax)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    spe_zoom_path = out_dir / f"spe_zoom_{far_label}_{t_start}_{t_end}.png"
    fig.savefig(spe_zoom_path, dpi=300)
    plt.close(fig)
    print(f"📈 Zoomed SPE figure saved to: {spe_zoom_path}")

    # ------------------------------------------------------------------
    # Figure 2: alarm (attack) only
    # ------------------------------------------------------------------
    if aa is not None:
        fig2, ax3 = plt.subplots(1, 1, figsize=(12, 2.5))
        ax3.step(
            idx,
            aa,
            where="post",
            label="Alarm (attack)",
        )
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Alarm")
        ax3.set_xlabel("Time (samples)")
        ax3.legend(loc="upper right")
        fig2.tight_layout()
        alarm_zoom_path = out_dir / f"alarms_zoom_{far_label}_{t_start}_{t_end}.png"
        fig2.savefig(alarm_zoom_path, dpi=300)
        plt.close(fig2)
        print(f"📈 Zoomed alarm figure saved to: {alarm_zoom_path}")


def build_apply_delta_step():
    """
    Renvoie une fonction apply_delta_step(x_t, S, u, alpha_t) conforme à l’équation
    de manipulation : on perturbe seulement les canaux S avec un signe u_j et
    une amplitude alpha_t (partagée équitablement).
    """

    def apply_delta_step(x_t_np: np.ndarray, S: np.ndarray, u: np.ndarray, alpha_t: float) -> np.ndarray:
        x = np.asarray(x_t_np, dtype=np.float32).copy()
        if alpha_t == 0.0 or len(S) == 0:
            return x
        share = 1.0 / float(len(S))
        for idx, ch in enumerate(S):
            x[ch] += alpha_t * u[idx] * share
        return x

    return apply_delta_step


def sweep_vacf_grid(
        X_fault_free: np.ndarray,
        spe_fn,
        apply_delta_step,
        J_SPE: float,
        base_cfg: ChatteringConfig,
        eps_list: list[float],
        K_list: list[float],
        out_dir: Path,
        rng_seed: int = 123,
) -> None:
    """
    Balaye une grille (eps_max, K) pour valider la théorie de manipulation de FAR.

    Pour chaque couple (eps_max, K), on :
      - construit un ChatteringConfig,
      - lance run_vacf_fault_free_experiment,
      - enregistre les métriques dans un CSV,
      - génère des courbes FAR_attack_emp vs eps_max (par valeur de K).

    X_fault_free : série concaténée *sans faute* (tous IDV=1..21 pré-fautes).
    spe_fn       : fonction x -> SPE(x) basée sur le DAE–PCA.
    apply_delta_step : injection d’un pas de perturbation (Δ).
    J_SPE        : seuil SPE.
    base_cfg     : config de base (on copie tout sauf eps_max et K).
    eps_list     : liste de valeurs ε_max à balayer.
    K_list       : liste de valeurs K (gain) à balayer.
    out_dir      : dossier de sortie pour CSV + figures.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vacf_sweep_eps_K.csv"

    rows = []

    # NOTE: For a fixed K, we reset the RNG for every eps so that the
    # same randomness (gate/S/u draws) is used across eps values.
    # This reduces finite-sample variability and makes FAR-vs-eps trends
    # easier to interpret.
    with open(csv_path, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "eps_max",
            "K",
            "FAR_base",
            "FAR_attack_emp",
            "FAR_attack_exact",
            "FAR_attack_approx",
            "APM_base",
            "APM_attack_emp",
            "APM_attack_exact",
            "APM_attack_approx",
            "rho_emp",
            "p_on_emp",
            "p_base_cond",
        ])

        for eps in eps_list:
            for K in K_list:
                cfg = ChatteringConfig(
                    k=base_cfg.k,
                    eps_max=eps,
                    L_on=base_cfg.L_on,
                    L_off=base_cfg.L_off,
                    K=K,
                    c_share=base_cfg.c_share,
                    fs_per_min=base_cfg.fs_per_min,
                    regime=base_cfg.regime,
                    gate_mode=getattr(base_cfg, "gate_mode", "bernoulli"),
                    rho=base_cfg.rho,
                    kc_consecutive=base_cfg.kc_consecutive,
                )

                # Deterministic RNG per K (reset for each eps)

                seed_K = int(rng_seed + round(1000.0 * float(K)))

                rng_point = np.random.default_rng(seed_K)

                res = run_vacf_fault_free_experiment(
                    X=X_fault_free,
                    spe_fn=spe_fn,
                    apply_delta_step=apply_delta_step,
                    J_SPE=J_SPE,
                    cfg=cfg,
                    rng=rng_point,
                    debug_dir=str(out_dir / f"eps{eps:.3f}_K{K:.3f}").replace(".", "p"),
                )

                # Clés retournées par exp_harness.run_vacf_fault_free_experiment
                FAR_base = float(res["FAR_base"])
                FAR_emp = float(res["FAR_attack_emp"])
                FAR_exact = float(res["FAR_attack_pred_exact"])
                FAR_approx = float(res["FAR_attack_pred_approx"])
                rho_emp = float(res["rho_emp"])
                p_on_emp = float(res["p_on_emp"])
                p_base_cond = float(res["p_base_cond"])
                fs = float(res["cfg"].fs_per_min)

                APM_base = FAR_base * fs
                APM_emp = FAR_emp * fs
                APM_exact = FAR_exact * fs
                APM_approx = FAR_approx * fs

                writer.writerow([
                    eps,
                    K,
                    FAR_base,
                    FAR_emp,
                    FAR_exact,
                    FAR_approx,
                    APM_base,
                    APM_emp,
                    APM_exact,
                    APM_approx,
                    rho_emp,
                    p_on_emp,
                    p_base_cond,
                ])

                rows.append(dict(
                    eps=eps,
                    K=K,
                    FAR_base=FAR_base,
                    FAR_emp=FAR_emp,
                    FAR_exact=FAR_exact,
                    FAR_approx=FAR_approx,
                ))

    print(f"✅ Sweep VACF enregistré dans : {csv_path}")

    # ---------------------- Figures pour le papier ----------------------
    # 1) FAR_attack_emp vs eps_max, courbes séparées par K
    by_K: dict[float, dict[str, list[float]]] = {}
    for row in rows:
        K = row["K"]
        if K not in by_K:
            by_K[K] = {"eps": [], "FAR_emp": [], "FAR_exact": [], "FAR_approx": []}
        by_K[K]["eps"].append(row["eps"])
        by_K[K]["FAR_emp"].append(row["FAR_emp"])
        by_K[K]["FAR_exact"].append(row["FAR_exact"])
        by_K[K]["FAR_approx"].append(row["FAR_approx"])

    plt.figure()
    for K, d in sorted(by_K.items(), key=lambda kv: kv[0]):
        # trier par eps croissant pour des courbes propres
        eps_arr, far_arr = zip(*sorted(zip(d["eps"], d["FAR_emp"])))
        plt.plot(eps_arr, far_arr, marker="o", label=f"K={K}")

    # baseline théorique (sans attaque) = FAR_base moyen
    if rows:
        far_base_mean = np.mean([r["FAR_base"] for r in rows])
        plt.axhline(far_base_mean, linestyle="--", label="FAR_base (mean)")

    plt.xlabel(r"$\varepsilon_{\max}$")
    plt.ylabel("FAR_attack (empirical)")
    plt.title("FAR_attack vs $\\varepsilon_{\\max}$ for different $K$")
    plt.legend()
    fig1_path = out_dir / "far_emp_vs_eps_by_K.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"📈 Figure sauvegardée : {fig1_path}")

    # 2) Comparaison FAR_emp vs FAR_exact (Eq.(20)) en scatter
    plt.figure()
    far_emp_all = [r["FAR_emp"] for r in rows]
    far_exact_all = [r["FAR_exact"] for r in rows]
    plt.scatter(far_exact_all, far_emp_all)
    min_val = min(far_emp_all + far_exact_all)
    max_val = max(far_emp_all + far_exact_all)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", label="y=x")
    plt.xlabel("FAR_attack (theory, Eq.(20))")
    plt.ylabel("FAR_attack (empirical)")
    plt.title("Validation of Eq.(20): empirical vs theoretical FAR")
    plt.legend()
    fig2_path = out_dir / "far_emp_vs_far_exact.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"📈 Figure sauvegardée : {fig2_path}")


def search_eps_for_target_far(
        X_fault_free: np.ndarray,
        spe_fn,
        apply_delta_step,
        J_SPE: float,
        base_cfg: ChatteringConfig,
        target_far: float,
        eps_min: float = 0.0,
        eps_max: float = 1.0,
        tol: float = 0.002,
        max_iter: int = 12,
        rng_seed: int = 999,
) -> Dict[str, Any]:
    """
    Cherche un ε_max tel que FAR_attack_emp ≈ target_far
    (régime fault-free, VACF) par dichotomie sur ε_max.

    - X_fault_free : série concaténée *sans faute* (comme X_normal_np)
    - spe_fn       : fonction x -> SPE(x) basée sur le DAE–PCA
    - apply_delta_step : injection d’un pas de perturbation (Δ)
    - J_SPE        : seuil SPE
    - base_cfg     : config de base (on garde k, ρ, L_on/L_off, etc.)
    - target_far   : FAR cible (ex. 0.10, 0.20)
    - eps_min/eps_max : borne inf/sup pour la recherche d'ε_max
    """

    def eval_eps(eps: float) -> Dict[str, Any]:
        # même config que base_cfg, mais avec eps_max modifié
        cfg = ChatteringConfig(
            k=base_cfg.k,
            eps_max=eps,
            L_on=base_cfg.L_on,
            L_off=base_cfg.L_off,
            K=base_cfg.K,
            c_share=base_cfg.c_share,
            fs_per_min=base_cfg.fs_per_min,
            regime=base_cfg.regime,
            gate_mode=base_cfg.gate_mode,
            rho=base_cfg.rho,
            kc_consecutive=base_cfg.kc_consecutive,
        )
        # on recrée un rng avec la même graine à chaque appel
        # → gate, S, u restent identiques ; seul ε_max change
        rng = np.random.default_rng(rng_seed)
        res = run_vacf_fault_free_experiment(
            X=X_fault_free,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            cfg=cfg,
            rng=rng,
            debug_dir=None,  # pas besoin d'artefacts ici
        )
        return res

    # 1) évaluer aux bornes
    res_lo = eval_eps(eps_min)
    res_hi = eval_eps(eps_max)
    far_lo = float(res_lo["FAR_attack_emp"])
    far_hi = float(res_hi["FAR_attack_emp"])
    far_base = float(res_lo["FAR_base"])

    # si même à eps_max on n’atteint pas la FAR cible → pas atteignable
    if far_hi < target_far:
        return {
            "reachable": False,
            "reason": "target_far above max FAR for eps_max",
            "eps_min": eps_min,
            "eps_max": eps_max,
            "far_lo": far_lo,
            "far_hi": far_hi,
            "far_base": far_base,
        }

    lo_eps, hi_eps = eps_min, eps_max
    best_res = res_hi
    best_eps = eps_max
    best_far = far_hi

    # 2) boucle de dichotomie
    for _ in range(max_iter):
        mid = 0.5 * (lo_eps + hi_eps)
        res_mid = eval_eps(mid)
        far_mid = float(res_mid["FAR_attack_emp"])

        # si on est assez proche de la cible → on s’arrête
        if abs(far_mid - target_far) < tol:
            best_res = res_mid
            best_eps = mid
            best_far = far_mid
            break

        # mise à jour de l’intervalle [lo, hi]
        if far_mid < target_far:
            lo_eps, far_lo = mid, far_mid
        else:
            hi_eps, far_hi = mid, far_mid

        # garder la meilleure approximation rencontrée
        if abs(far_mid - target_far) < abs(best_far - target_far):
            best_res = res_mid
            best_eps = mid
            best_far = far_mid

    # 3) paquet de résultats à renvoyer
    return {
        "reachable": True,
        "target_far": target_far,
        "eps_max": best_eps,
        # on renomme proprement ici pour coller à ce que tu affiches dans main()
        "FAR_emp": best_far,
        "FAR_exact": float(best_res["FAR_attack_pred_exact"]),
        "FAR_approx": float(best_res["FAR_attack_pred_approx"]),
        "FAR_base": far_base,
        "rho_emp": float(best_res["rho_emp"]),
        "p_on_emp": float(best_res["p_on_emp"]),
        "p_base_cond": float(best_res["p_base_cond"]),
    }


def run_vacf_on_idv_pre_fault(
        idv: int,
        opt,
        cfg: ChatteringConfig,
        spe_fn,
        apply_delta_step,
        J_SPE: float,
        out_dir: Path,
        rng_seed: int = 1234,
):
    """
    Run the VACF chattering attack on the pre-fault window of a single TEP
    fault scenario (IDV). We reuse the same machinery as the fault-free VACF
    experiment (run_vacf_fault_free_experiment).
    """
    # 1) Charger les données de test pour ce scénario de faute
    #    → même pattern que PHASE 3A
    _, _, testloader = dataset(idv)
    test_x = _to_tensor(testloader)  # shape (N, n_features)

    # 2) Garder uniquement la fenêtre fault-free [0 .. f-1]
    n_pre = int(opt.f)
    n_pre = min(n_pre, test_x.shape[0])
    X_pre = test_x[:n_pre].cpu().numpy().astype(np.float32)  # (n_pre, n_features)

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(rng_seed)

    # 3) Lancer exactement la même expérience qu'en fault-free global,
    #    mais sur cet IDV seulement.
    results = run_vacf_fault_free_experiment(
        X=X_pre,
        spe_fn=spe_fn,
        apply_delta_step=apply_delta_step,
        J_SPE=J_SPE,
        cfg=cfg,
        rng=rng,
        debug_dir=str(out_dir),
    )

    # results contient FAR_base, FAR_attack_emp, delta_FAR_emp, etc.
    # out_dir contient aussi series.npz pour la visualisation.
    return results


# ============================================================
#  MAIN : pipeline complet + expérience VACF fault-free
# ============================================================

def main():
    # Seeds pour reproductibilité
    np.random.seed(42)
    t.manual_seed(42)

    # -------------------------------
    # PHASE 1 : Entraînement DAE-PCA
    # -------------------------------
    print("\n🚀 PHASE 1 : Entraînement du modèle DAE-PCA")
    print("Entraînement sur le cas normal (IDV=1)")

    trainloader, valloader, _ = dataset(1)  # IDV(1) = données normales
    train_x = _to_tensor(trainloader)
    val_x = _to_tensor(valloader)

    Ae = AE(opt)

    min_loss = float('inf')
    best_epoch = -1
    best_path = os.path.join(opt.model_state_path, "best.pth.tar")
    os.makedirs(opt.model_state_path, exist_ok=True)

    for epoch in range(opt.max_epoch):
        Ae, train_loss, l1, l2 = train(Ae, train_x, epoch)
        val_loss = val(Ae, val_x)

        if val_loss < min_loss:
            min_loss = val_loss
            best_epoch = epoch
            t.save(Ae.state_dict(), best_path)

        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{opt.max_epoch}] | "
                f"Tr: {train_loss:.6f} | Va: {val_loss:.6f} | "
                f"L1: {l1:.6f} | L2: {l2:.6f}"
            )

    print(f"✅ Meilleur modèle sauvegardé à l'époque {best_epoch}, Val Loss: {min_loss:.6f}")

    # ----------------------------------------------
    # PHASE 2 : Calcul des seuils T² et SPE (normal)
    # ----------------------------------------------
    print("\n📊 PHASE 2 : Calcul des seuils de détection")

    T2c, SPEc, cov_T, T_train = thre(Ae, best_path, train_x)
    J_T2 = calculate_threshold(T2c, alpha=0.99)  # Seuil T² (99e percentile)
    J_SPE = calculate_threshold(SPEc, alpha=0.99)  # Seuil SPE (99e percentile)

    sigma_SPE_X0 = float(np.std(SPEc, ddof=1))  # std on fault-free validation (X0 split)

    print(f"Seuils calculés -> T²: {J_T2:.4f}, SPE: {J_SPE:.4f}")

    # Sauvegarder les seuils pour les scripts de génération de figures
    os.makedirs("runs", exist_ok=True)
    np.savez("runs/thresholds.npz", J_T2=J_T2, J_SPE=J_SPE, sigma_SPE_X0=sigma_SPE_X0)

    # ----------------------------------------------------------------------
    # PHASE 3A : Baseline (sans attaque) sur TOUTES les fautes IDV=1..21
    # ----------------------------------------------------------------------
    print("\n📉 PHASE 3A : Baseline (sans attaque) sur toutes les fautes IDV(1..21)")

    baseline_results: Dict[int, Dict[str, Any]] = {}
    os.makedirs("runs", exist_ok=True)
    f_win = int(opt.f)

    for idv in range(1, 22):
        _, _, testloader = dataset(idv)
        test_x = _to_tensor(testloader)

        T2, SPE, BIC = tes(best_path, test_x, cov_T, J_T2, J_SPE)
        metrics = eval_far_fdr(T2, SPE, BIC, f_win, J_T2, J_SPE)
        baseline_results[idv] = metrics

        print(
            f"  IDV={idv:2d} | "
            f"FAR_SPE={metrics['FAR_SPE']:.4f} | FDR_SPE={metrics['FDR_SPE']:.4f} | "
            f"FAR_T2={metrics['FAR_T2']:.4f} | FDR_T2={metrics['FDR_T2']:.4f}"
        )

    baseline_path = os.path.join("runs", "baseline_all_idv.npy")
    np.save(baseline_path, baseline_results, allow_pickle=True)
    print(f"\n✅ Résultats baseline (tous IDV) sauvegardés dans : {baseline_path}")

    # ----------------------------------------------------------------------
    # PHASE 3B : Préparation de SPE(x) pour l'attaque fault-free
    # ----------------------------------------------------------------------
    print("\n⚙️ PHASE 3B : Préparation de la fonction SPE(x) pour l'attaque fault-free")
    spe_fn = build_spe_fn(best_path)
    apply_delta_step = build_apply_delta_step()

    # ----------------------------------------------------------------------
    # PHASE 3C : Expérience chattering FDIA en régime fault-free (VACF)
    # ----------------------------------------------------------------------
    print("\n🔥 PHASE 3C : Expérience chattering FDIA en régime fault-free (VACF)")

    f_index = int(opt.f)

    # On construit un long signal "fault-free" en concaténant les segments [0,f)
    # de tous les IDV (pré-fautes).
    segs = []
    IDV_LIST = list(range(1, 22))
    for I in IDV_LIST:
        _, _, Xte = dataset(I)
        Xte_t = _to_tensor(Xte)
        X_pre = Xte_t[:f_index]  # [0, f)
        segs.append(X_pre)

    X_normal = t.cat(segs, dim=0)
    X_normal_np = X_normal.cpu().numpy().astype(np.float32)
    print(f"[VACF] Données fault-free concaténées : shape = {X_normal_np.shape}")

    # Paramètres d'attaque
    # Option B (gate périodique) : duty cycle ρ = L_on/(L_on+L_off).
    # Ici: ρ = 30/(30+15) ≈ 0.6667.
    vacf_cfg = ChatteringConfig(
        k=3,
        eps_max=0.15,
        L_on=30,
        L_off=15,
        K=5.5,  # pas utilisé dans cette version, mais gardé pour compat
        c_share=1.0,
        fs_per_min=1.0,  # fréquence d'échantillonnage (échantillons/minute)
        regime="per-sample",
        gate_mode="periodic",
        rho=30 / (30 + 15),
        kc_consecutive=1,
    )

    vacf_dir = Path("runs/vacf_far3")
    vacf_dir.mkdir(parents=True, exist_ok=True)

    vacf_res = run_vacf_fault_free_experiment(
        X=X_normal_np,
        spe_fn=spe_fn,
        apply_delta_step=apply_delta_step,
        J_SPE=J_SPE,
        cfg=vacf_cfg,
        rng=np.random.default_rng(123),
        debug_dir=str(vacf_dir),
    )

    FAR_base = vacf_res["FAR_base"]
    FAR_emp = vacf_res["FAR_attack_emp"]
    FAR_exact = vacf_res["FAR_attack_pred_exact"]
    FAR_approx = vacf_res["FAR_attack_pred_approx"]
    delta_emp = vacf_res["delta_FAR_emp"]
    delta_exact = vacf_res["delta_FAR_pred_exact"]
    delta_approx = vacf_res["delta_FAR_pred_approx"]

    rho_emp = vacf_res["rho_emp"]
    p_on_emp = vacf_res["p_on_emp"]
    p_base_cond = vacf_res["p_base_cond"]

    fs = vacf_res["cfg"].fs_per_min
    APM_base = fs * FAR_base
    APM_emp = fs * FAR_emp

    print("\n[VACF fault-free experiment]")
    print(f"Baseline FAR           = {FAR_base:.4f}  (APM_base = {APM_base:.4f})")
    print(f"Attack FAR (empirique) = {FAR_emp:.4f}  (APM_attack = {APM_emp:.4f}, Δ_emp = {delta_emp:.4f})")
    print(f"Attack FAR (théorie, Eq.(20)) = {FAR_exact:.4f}  (Δ_exact = {delta_exact:.4f})")
    print(f"Attack FAR (approx, Eq.(21))  = {FAR_approx:.4f}  (Δ_approx = {delta_approx:.4f})")

    print("\nDétails VACF :")
    print(f"  ρ_emp       = {rho_emp:.4f}   (gate 'ON' empirique)")
    print(f"  p_on_emp    = {p_on_emp:.4f}   (P[SPE>J | gate=1] sous attaque)")
    print(f"  p_base_cond = {p_base_cond:.4f} (P_base[SPE>J | gate=1])")
    print("\n✅ Artefacts de l'expérience sauvegardés dans : runs/vacf_fault_free")

    # ------------------------------------------------------------------
    # PHASE 3D : Visualisation SPE + alarmes pour l’article
    # ------------------------------------------------------------------
    print("\n📊 PHASE 3D : Visualisation SPE + alarmes ")
    plot_vacf_timeseries(vacf_dir, J_SPE)

    # =========================== PHASE 3D : Sweep εmax, K ===========================
    print("\n📌 PHASE 3D : Sweep automatique sur eps_max et K (validation théorie VACF)")

    # 1) Données fault-free globales (celles construites juste avant pour VACF)
    #    On réutilise X_normal_np qui concatène les fenêtres pré-fautes de plusieurs IDV.
    X_fault_free = X_normal_np

    # 2) Config de base : on part de vacf_cfg (celle utilisée en PHASE 3C)
    base_cfg = vacf_cfg

    # 3) Listes de paramètres à explorer pour le papier
    eps_list = [0.05, 0.10, 0.15, 0.20]
    K_list = [0.5, 1.0, 2.0]

    sweep_dir = Path("runs/vacf_sweep_eps_K")

    sweep_vacf_grid(
        X_fault_free=X_fault_free,
        spe_fn=spe_fn,
        apply_delta_step=apply_delta_step,
        J_SPE=J_SPE,
        base_cfg=base_cfg,
        eps_list=eps_list,
        K_list=K_list,
        out_dir=sweep_dir,
        rng_seed=123,
    )

    # -------------------------------------------------------------------------
    # PHASE 3E : Search for ε_max for target FAR levels (10 %, 20 %, 45 %)
    # -------------------------------------------------------------------------
    print("\n🎯 PHASE 3E : Search for ε_max for target FAR levels (10 %, 20 %, 45 %)")

    target_FAR_list = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    target_cfgs = {}  # pour mémoriser les configs d'attaque par FAR cible

    for target_far in target_FAR_list:
        # Use a dedicated seed per target FAR for reproducibility
        rng_seed = 999 + int(100 * target_far)

        # 1) Find ε_max such that FAR_attack_emp ≈ target_far
        res_target = search_eps_for_target_far(
            X_fault_free=X_fault_free,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            base_cfg=vacf_cfg,
            target_far=target_far,
            rng_seed=rng_seed,
        )

        if not res_target["reachable"]:
            print(
                f"\n  Target FAR={target_far * 100:.1f}%  →  not reachable in the explored ε_max range."
            )
            continue

        eps_star = float(res_target["eps_max"])
        far_emp_est = float(res_target["FAR_emp"])
        far_exact_est = float(res_target["FAR_exact"])
        print(
            f"\n  Target FAR={target_far * 100:.1f}%  →  ε_max≈{eps_star:.4f}, "
            f"FAR_emp(est)={far_emp_est:.4f}, FAR_exact(est)={far_exact_est:.4f}"
        )

        # 2) Run a FULL VACF experiment with this ε_max*, to validate in practice
        cfg_target = ChatteringConfig(
            k=vacf_cfg.k,
            eps_max=eps_star,
            L_on=vacf_cfg.L_on,
            L_off=vacf_cfg.L_off,
            K=vacf_cfg.K,
            c_share=vacf_cfg.c_share,
            fs_per_min=vacf_cfg.fs_per_min,
            regime=vacf_cfg.regime,
            gate_mode=vacf_cfg.gate_mode,
            rho=vacf_cfg.rho,
            kc_consecutive=vacf_cfg.kc_consecutive,
        )

        # mémoriser la config associée à cette FAR cible
        target_cfgs[target_far] = cfg_target

        tag_short = f"{int(round(100 * target_far)):02d}"
        target_dir = Path(f"runs/vacf_far{tag_short}_target")
        target_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  ▶ Running full VACF experiment for target FAR {target_far * 100:.1f}%")
        full_res = run_vacf_fault_free_experiment(
            X=X_fault_free,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            cfg=cfg_target,
            rng=np.random.default_rng(rng_seed),
            debug_dir=str(target_dir),
        )

        # 3) Print detailed metrics (same style as PHASE 3C)
        FAR_base_full = float(full_res["FAR_base"])
        FAR_emp_full = float(full_res["FAR_attack_emp"])
        FAR_exact_full = float(full_res["FAR_attack_pred_exact"])
        FAR_approx_full = float(full_res["FAR_attack_pred_approx"])
        delta_emp_full = float(full_res["delta_FAR_emp"])
        delta_exact_full = float(full_res["delta_FAR_pred_exact"])
        delta_approx_full = float(full_res["delta_FAR_pred_approx"])
        rho_emp_full = float(full_res["rho_emp"])
        p_on_emp_full = float(full_res["p_on_emp"])
        p_base_cond_full = float(full_res["p_base_cond"])
        fs_full = float(full_res["cfg"].fs_per_min)
        APM_base_full = fs_full * FAR_base_full
        APM_emp_full = fs_full * FAR_emp_full

        print(f"\n  [VACF run for target FAR {target_far * 100:.1f}%]")
        print(f"    FAR_base           = {FAR_base_full:.4f}  (APM_base = {APM_base_full:.4f})")
        print(
            f"    FAR_attack_emp     = {FAR_emp_full:.4f}  "
            f"(APM_attack = {APM_emp_full:.4f}, Δ_emp = {delta_emp_full:.4f}, "
            f"FAR_target = {target_far:.4f})"
        )
        print(
            f"    FAR_attack_exact   = {FAR_exact_full:.4f}  "
            f"(Δ_exact = {delta_exact_full:.4f})"
        )
        print(
            f"    FAR_attack_approx  = {FAR_approx_full:.4f}  "
            f"(Δ_approx = {delta_approx_full:.4f})"
        )
        print("    VACF details (target run) :")
        print(f"      ρ_emp       = {rho_emp_full:.4f}")
        print(f"      p_on_emp    = {p_on_emp_full:.4f}")
        print(f"      p_base_cond = {p_base_cond_full:.4f}")

        # 4) Figures for the paper:
        #    (a) full-length baseline vs attack,
        #    (b) zoomed SPE around J_SPE,
        #    (c) separate alarm timeline for the same zoom window
        plot_vacf_timeseries(target_dir, J_SPE)
        plot_vacf_timeseries_zoom(
            target_dir,
            J_SPE,
            t_start=400,
            t_end=800,
            y_max_cap=0.5,  # zoom vertical pour mieux voir autour du seuil
        )

    # -------------------------------------------------------------------------
    # PHASE 4A : Case studies on specific TEP faults (IDV 1 TO IDV 21)
    # -------------------------------------------------------------------------
    print("\n📊 PHASE 4A : IDV-specific case studies under chattering FDIA for General FAR Target 10%")

    # Si la recherche n'a pas trouvé 10 %, on retombe sur la config de base.
    cfg_idv = target_cfgs.get(0.10, vacf_cfg)

    print("\n  [VACF configuration used for IDV case studies]")
    print(f"    (Nominal target FAR ≈ 10%)")
    print(f"    k       = {cfg_idv.k:d}")
    print(f"    eps_max = {cfg_idv.eps_max:.4f}")
    print(f"    rho     = {cfg_idv.rho:.2f}")
    print(f"    L_on    = {cfg_idv.L_on:d}")
    print(f"    L_off   = {cfg_idv.L_off:d}")
    print(f"    fs      = {cfg_idv.fs_per_min:.2f} samples/min\n")

    for idv in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        idv_dir = Path(f"runs/A_vacf_idv{idv}")
        print(f"  ▶ Running VACF attack on pre-fault window of IDV={idv} \n")

        res_idv = run_vacf_on_idv_pre_fault(
            idv=idv,
            opt=opt,
            cfg=cfg_idv,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            out_dir=idv_dir,
            rng_seed=3000 + idv,
        )

        FAR_base_idv = float(res_idv["FAR_base"])
        FAR_emp_idv = float(res_idv["FAR_attack_emp"])
        delta_idv = float(res_idv["delta_FAR_emp"])

        # Config effectivement utilisée (au cas où on modifie run_vacf plus tard)
        cfg_used = res_idv["cfg"]

        print(
            f"    [IDV {idv}] FAR_base_pre = {FAR_base_idv:.4f}, "
            f"FAR_attack_pre = {FAR_emp_idv:.4f}, ΔFAR_pre = {delta_idv:.4f}"
        )
        print(
            "            (used VACF config: "
            f"k={cfg_used.k:d}, eps_max={cfg_used.eps_max:.4f}, "
            f"rho={cfg_used.rho:.2f}, L_on={cfg_used.L_on:d}, "
            f"L_off={cfg_used.L_off:d})"
        )

        # Figures pour cet IDV : même style que pour le cas fault-free global
        plot_vacf_timeseries(idv_dir, J_SPE)

        # Zoom sur toute la fenêtre fault-free [0 .. f-1]
        plot_vacf_timeseries_zoom(
            idv_dir,
            J_SPE,
            t_start=0,
            t_end=int(opt.f),  # sera clampé à la longueur réelle dans la fonction
            y_max_cap=None,  # pas de zoom vertical forcé ici
        )

    # -------------------------------------------------------------------------
    # PHASE 4B : Case studies on specific TEP faults (IDV 1 to IDV 21)
    # ---------------------------------------------------------------------
    print("\n📊 PHASE 4B : IDV-specific case studies under chattering FDIA for General FAR Target 20%")

    cfg_idv = target_cfgs.get(0.20, vacf_cfg)

    print("\n  [VACF configuration used for IDV case studies]")
    print(f"    (Nominal target FAR ≈ 20%)")
    print(f"    k       = {cfg_idv.k:d}")
    print(f"    eps_max = {cfg_idv.eps_max:.4f}")
    print(f"    rho     = {cfg_idv.rho:.2f}")
    print(f"    L_on    = {cfg_idv.L_on:d}")
    print(f"    L_off   = {cfg_idv.L_off:d}")
    print(f"    fs      = {cfg_idv.fs_per_min:.2f} samples/min\n")

    for idv in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        idv_dir = Path(f"runs/B_vacf_idv{idv}")
        print(f"  ▶ Running VACF attack on pre-fault window of IDV={idv} \n")

        res_idv = run_vacf_on_idv_pre_fault(
            idv=idv,
            opt=opt,
            cfg=cfg_idv,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            out_dir=idv_dir,
            rng_seed=3000 + idv,
        )

        FAR_base_idv = float(res_idv["FAR_base"])
        FAR_emp_idv = float(res_idv["FAR_attack_emp"])
        delta_idv = float(res_idv["delta_FAR_emp"])

        # Config effectivement utilisée (au cas où on modifie run_vacf plus tard)
        cfg_used = res_idv["cfg"]

        print(
            f"\n    [IDV {idv}] FAR_base_pre = {FAR_base_idv:.4f}, "
            f"FAR_attack_pre = {FAR_emp_idv:.4f}, ΔFAR_pre = {delta_idv:.4f}"
        )
        print(
            "            (used VACF config: "
            f"k={cfg_used.k:d}, eps_max={cfg_used.eps_max:.4f}, "
            f"rho={cfg_used.rho:.2f}, L_on={cfg_used.L_on:d}, "
            f"L_off={cfg_used.L_off:d})"
        )

        # Figures pour cet IDV : même style que pour le cas fault-free global
        plot_vacf_timeseries(idv_dir, J_SPE)

        # Zoom sur toute la fenêtre fault-free [0 .. f-1]
        plot_vacf_timeseries_zoom(
            idv_dir,
            J_SPE,
            t_start=0,
            t_end=int(opt.f),  # sera clampé à la longueur réelle dans la fonction
            y_max_cap=None,  # pas de zoom vertical forcé ici
        )

    # -------------------------------------------------------------------------
    # PHASE 4C : Case studies on specific TEP faults (IDV 4 and IDV 9)
    # -------------------------------------------------------------------------
    print("\n📊 PHASE 4C : IDV-specific case studies under chattering FDIA for General FAR Target 40%")

    cfg_idv = target_cfgs.get(0.40, vacf_cfg)

    print("\n  [VACF configuration used for IDV case studies]")
    print(f"    (Nominal target FAR ≈ 40%)")
    print(f"    k       = {cfg_idv.k:d}")
    print(f"    eps_max = {cfg_idv.eps_max:.4f}")
    print(f"    rho     = {cfg_idv.rho:.2f}")
    print(f"    L_on    = {cfg_idv.L_on:d}")
    print(f"    L_off   = {cfg_idv.L_off:d}")
    print(f"    fs      = {cfg_idv.fs_per_min:.2f} samples/min\n")

    for idv in [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        idv_dir = Path(f"runs/C_vacf_idv{idv}")
        print(f"  ▶ Running VACF attack on pre-fault window of IDV={idv} \n")

        res_idv = run_vacf_on_idv_pre_fault(
            idv=idv,
            opt=opt,
            cfg=cfg_idv,
            spe_fn=spe_fn,
            apply_delta_step=apply_delta_step,
            J_SPE=J_SPE,
            out_dir=idv_dir,
            rng_seed=3000 + idv,
        )

        FAR_base_idv = float(res_idv["FAR_base"])
        FAR_emp_idv = float(res_idv["FAR_attack_emp"])
        delta_idv = float(res_idv["delta_FAR_emp"])

        # Config effectivement utilisée (au cas où on modifie run_vacf plus tard)
        cfg_used = res_idv["cfg"]

        print(
            f"\n    [IDV {idv}] FAR_base_pre = {FAR_base_idv:.4f}, "
            f"FAR_attack_pre = {FAR_emp_idv:.4f}, ΔFAR_pre = {delta_idv:.4f}"
        )
        print(
            "            (used VACF config: "
            f"k={cfg_used.k:d}, eps_max={cfg_used.eps_max:.4f}, "
            f"rho={cfg_used.rho:.2f}, L_on={cfg_used.L_on:d}, "
            f"L_off={cfg_used.L_off:d})"
        )

        # Figures pour cet IDV : même style que pour le cas fault-free global
        plot_vacf_timeseries(idv_dir, J_SPE)

        # Zoom sur toute la fenêtre fault-free [0 .. f-1]
        plot_vacf_timeseries_zoom(
            idv_dir,
            J_SPE,
            t_start=0,
            t_end=int(opt.f),  # sera clampé à la longueur réelle dans la fonction
            y_max_cap=None,  # pas de zoom vertical forcé ici
        )


if __name__ == "__main__":
    main()


