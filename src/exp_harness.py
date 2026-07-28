# -*- coding: utf-8 -*-
"""
exp_harness.py — VACF fault-free chattering FDIA

Cette version implémente uniquement :
  - un gate Bernoulli(ρ) sur toute la trajectoire (régime sans faute),
  - une attaque chattering simple : perturbation constante eps_max
    sur un sous-ensemble de k capteurs,
  - le mapping FAR ↔ (ρ, p_on) via les équations (20)–(21).

Interface principale :
    run_vacf_fault_free_experiment(X, spe_fn, apply_delta_step, J_SPE, cfg, rng, debug_dir)
"""

from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional

import numpy as np
import os
import json
from pathlib import Path


@dataclass
class ChatteringConfig:
    # paramètres "géométriques" de l'attaque
    k: int = 3              # nombre de capteurs attaqués
    eps_max: float = 0.05   # amplitude maximale de la perturbation
    # Paramètres d'un gate périodique (si gate_mode="periodic")
    L_on: int = 3
    L_off: int = 7
    K: float = 0.0          # non utilisé dans cette version (pas de feedback)
    c_share: float = 1.0    # partage de l'amplitude entre les canaux
    # timing / VACF
    fs_per_min: float = 1.0 # fréquence d'échantillonnage (samples / minute)
    regime: str = "per-sample"
    # Gate activation.
    # - gate_mode="bernoulli": P(gate(t)=1)=rho (i.i.d.)
    # - gate_mode="periodic" : rho est déduit comme L_on/(L_on+L_off)
    rho: float = 0.4

    # Mode de gate temporel: "bernoulli" (i.i.d.) ou "periodic" (déterministe)
    gate_mode: str = "bernoulli"
    gate_mode: str = "bernoulli"  # "bernoulli" | "periodic"
    kc_consecutive: int = 1 # politique d'alarmes consécutives (non utilisée ici)


def _ensure_rho_fraction(rho: float) -> float:
    """
    Si rho ressemble à un pourcentage (>1 et <=100), on le convertit en fraction.
    Sinon on le laisse tel quel.
    """
    if rho > 1.0 and rho <= 100.0:
        print(f"[warn] rho={rho} ressemble à un pourcentage → utilisation rho/100.")
        return rho / 100.0
    return float(rho)


def run_vacf_fault_free_experiment(
    X: np.ndarray,
    spe_fn: Callable[[np.ndarray], float],
    apply_delta_step: Callable[[np.ndarray, np.ndarray, np.ndarray, float], np.ndarray],
    J_SPE: float,
    cfg: ChatteringConfig,
    rng: Optional[np.random.Generator] = None,
    debug_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Expérience VACF en régime fault-free.

    Entrées
    -------
    X : np.ndarray (N, D)
        Trajectoire multi-variée en régime sans faute (concatenation des pré-fautes).
    spe_fn : callable
        Fonction SPE(x) -> float, définie côté main à partir de l'AE.
    apply_delta_step : callable
        (x_t, S, u, alpha_t) -> x_t_perturbé
    J_SPE : float
        Seuil du SPE (99e percentile typiquement).
    cfg : ChatteringConfig
        Hyperparamètres de l'attaque.
    rng : np.random.Generator ou None
        Générateur aléatoire, pour reproductibilité.
    debug_dir : str ou None
        Si non None, sauvegarde des artefacts (npz + metrics.json).

    Sortie
    ------
    dict contenant notamment :
        FAR_base, FAR_attack_emp,
        FAR_attack_pred_exact, FAR_attack_pred_approx,
        delta_FAR_emp, delta_FAR_pred_exact, delta_FAR_pred_approx,
        rho_emp, p_on_emp, p_base_cond, cfg, S, u, gate, ...
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X doit être 2D (N,D), reçu shape={X.shape}")

    N, D = X.shape
    if rng is None:
        rng = np.random.default_rng()

    # Gate sur tout l'horizon fault-free
    if cfg.gate_mode.lower() == "periodic":
        L_on = int(cfg.L_on)
        L_off = int(cfg.L_off)
        L = L_on + L_off
        if L_on <= 0 or L_off < 0 or L <= 0:
            raise ValueError(
                f"Invalid periodic gate params: L_on={cfg.L_on}, L_off={cfg.L_off} (need L_on>0, L_off>=0)"
            )
        rho = L_on / float(L)  # rho = L_on/(L_on+L_off)
        # Gate déterministe: 1 pendant L_on échantillons, puis 0 pendant L_off (répété)
        gate = ((np.arange(N) % L) < L_on).astype(np.int32)
        # Si un rho incohérent est passé via cfg.rho, on le signale (mais on garde la définition périodique)
        rho_in = _ensure_rho_fraction(cfg.rho)
        if abs(rho_in - rho) > 1e-6:
            print(f"[exp_harness] Warning: cfg.rho={rho_in:.4f} but periodic gate implies rho={rho:.4f} (L_on/L). Using rho={rho:.4f}.")
    else:
        # Gate Bernoulli(ρ) i.i.d.
        rho = _ensure_rho_fraction(cfg.rho)
        gate = rng.binomial(1, rho, size=N).astype(np.int32)  # gate(t)~Bernoulli(rho)

    # Sous-ensemble S de capteurs attaqués + signe u_j ∈ {−1, +1}
    k = max(1, min(int(cfg.k), D))
    S = rng.choice(D, size=k, replace=False)
    u = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=k)

    delta = np.zeros((N, D), dtype=np.float32)  # Δ(t, sensor) sur tous capteurs
    alpha_vec = np.zeros(N, dtype=np.float32)  # alpha(t)

    # Séries SPE et alarms (baseline vs attaque)
    SPE0 = np.zeros(N, dtype=np.float32)
    SPE1 = np.zeros(N, dtype=np.float32)
    alarms0 = np.zeros(N, dtype=np.int32)
    alarms1 = np.zeros(N, dtype=np.int32)

    # Boucle temporelle
    alpha = float(cfg.eps_max)  # amplitude constante = eps_max (pas de feedback K)

    # IMPORTANT : cohérence avec apply_delta_step de main.py
    # main.py fait: x[ch] += alpha_t * u[idx] * (1/len(S))
    share = (float(cfg.c_share) / float(k))  # si tu veux utiliser c_share, sinon 1.0/k

    for t in range(N):
        x_t = X[t]

        # Baseline
        spe0 = spe_fn(x_t)
        SPE0[t] = spe0
        alarms0[t] = 1 if spe0 > J_SPE else 0

        # Attaque chattering (si gate=1)
        if gate[t] == 1:
            alpha_vec[t] = alpha
            # Δ injecté seulement sur S
            delta[t, S] = alpha * u * share
            x_att = apply_delta_step(x_t, S, u, alpha_t=alpha)
        else:
            x_att = x_t

        spe1 = spe_fn(x_att)
        SPE1[t] = spe1
        alarms1[t] = 1 if spe1 > J_SPE else 0

    # FAR baseline / attaque (sur tout l'horizon fault-free)
    FAR_base = float(alarms0.mean())
    FAR_attack_emp = float(alarms1.mean())
    delta_FAR_emp = FAR_attack_emp - FAR_base

    # Paramètres empiriques du mapping VACF
    rho_emp = float(gate.mean())
    idx_on = gate == 1
    on_count = int(idx_on.sum())

    if on_count > 0:
        p_on_emp = float(((SPE1 > J_SPE) & idx_on).sum() / on_count)
        p_base_cond = float(((SPE0 > J_SPE) & idx_on).sum() / on_count)
    else:
        p_on_emp = 0.0
        p_base_cond = 0.0

    # Eq. (20) : FAR_exact = (1-ρ) p_base|gate + ρ p_on
    FAR_attack_pred_exact = (1.0 - rho_emp) * p_base_cond + rho_emp * p_on_emp

    # Eq. (21) : FAR_approx ≈ ρ p_on  (quand p_base|gate ≈ FAR_base est petit)
    FAR_attack_pred_approx = rho_emp * p_on_emp

    delta_FAR_pred_exact = FAR_attack_pred_exact - FAR_base
    delta_FAR_pred_approx = FAR_attack_pred_approx - FAR_base

    # Sauvegarde des artefacts (optionnel)
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        # Fichier de séries au format attendu par plot_vacf_timeseries() dans main.py
        np.savez(
            os.path.join(debug_dir, "series.npz"),
            X=X,
            S=S,
            u=u,
            gate=gate,
            SPE_base=SPE0,        # baseline
            SPE_attack=SPE1,      # sous attaque
            alarms_base=alarms0,  # alarmes baseline
            alarms_attack=alarms1, # alarmes attaque
            delta = delta,  # <-- AJOUT
            alpha = alpha_vec,  # <-- AJOUT
        )

        metrics: Dict[str, Any] = {
            "FAR_base": FAR_base,
            "FAR_attack_emp": FAR_attack_emp,
            "FAR_attack_pred_exact": FAR_attack_pred_exact,
            "FAR_attack_pred_approx": FAR_attack_pred_approx,
            "delta_FAR_emp": delta_FAR_emp,
            "delta_FAR_pred_exact": delta_FAR_pred_exact,
            "delta_FAR_pred_approx": delta_FAR_pred_approx,
            "rho_emp": rho_emp,
            "p_on_emp": p_on_emp,
            "p_base_cond": p_base_cond,
            "cfg": {
                "k": cfg.k,
                "eps_max": cfg.eps_max,
                "L_on": cfg.L_on,
                "L_off": cfg.L_off,
                "K": cfg.K,
                "c_share": cfg.c_share,
                "fs_per_min": cfg.fs_per_min,
                "regime": cfg.regime,
                "rho": cfg.rho,
                "kc_consecutive": cfg.kc_consecutive,
            },
        }

        with open(os.path.join(debug_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


    # Retour
    return {
        "cfg": cfg,
        "S": S,
        "u": u,
        "gate": gate,
        "SPE0": SPE0,
        "SPE1": SPE1,
        "alarms0": alarms0,
        "alarms1": alarms1,
        "FAR_base": FAR_base,
        "FAR_attack_emp": FAR_attack_emp,
        "FAR_attack_pred_exact": FAR_attack_pred_exact,
        "FAR_attack_pred_approx": FAR_attack_pred_approx,
        "delta_FAR_emp": delta_FAR_emp,
        "delta_FAR_pred_exact": delta_FAR_pred_exact,
        "delta_FAR_pred_approx": delta_FAR_pred_approx,
        "rho_emp": rho_emp,
        "p_on_emp": p_on_emp,
        "p_base_cond": p_base_cond,
    }

def apply_countermeasure_block(
    spe: np.ndarray,
    J: float,
    sigma_spe_x0: float,
    Kc: int = 2,
    L: int = 20,
    nu_max: int = 6,
    delta_factor: float = 0.5,
):
    """
    Offline causal simulation of the countermeasure block:

    1) Chattering-aware threshold adaptation:
       - Compute raw alarm a_t = 1[spe_t > J_eff(t)]
       - Maintain toggle count ν_t over a sliding window L using |a_t - a_{t-1}|
       - If ν_t > nu_max, raise threshold: J_eff = J + delta
         else J_eff = J
       (causal: decision affects current step using state from previous updates)

    2) Consecutive-Kc policy (delay timer):
       - Output y_t = 1 if we have >= Kc consecutive ones in the adapted raw alarm stream.

    Returns:
      alarms_def: defended alarm stream (0/1)
      alarms_adapted_raw: raw alarms after adaptive thresholding (before Kc)
      J_eff_series: effective threshold over time
      nu_series: toggle-rate indicator over time
    """
    spe = np.asarray(spe).astype(float).ravel()
    T = spe.shape[0]

    delta = float(delta_factor * sigma_spe_x0)

    alarms_adapted = np.zeros(T, dtype=np.int64)
    alarms_def = np.zeros(T, dtype=np.int64)
    J_eff_series = np.zeros(T, dtype=float)
    nu_series = np.zeros(T, dtype=np.int64)

    # Toggle-rate: store last L diffs |a_t - a_{t-1}|
    from collections import deque
    diffs = deque(maxlen=L)
    prev_a = 0
    consec = 0
    J_eff = J

    for t in range(T):
        # raw alarm under current effective threshold
        a = 1 if spe[t] > J_eff else 0
        alarms_adapted[t] = a

        # update toggle-rate window
        d = abs(a - prev_a)
        diffs.append(d)
        nu_t = int(sum(diffs))
        nu_series[t] = nu_t

        # update effective threshold for next samples (causal update)
        if nu_t > nu_max:
            J_eff = J + delta
        else:
            J_eff = J
        J_eff_series[t] = J_eff

        # consecutive-Kc policy on adapted raw alarms
        if a == 1:
            consec += 1
        else:
            consec = 0
        alarms_def[t] = 1 if consec >= Kc else 0

        prev_a = a

    return alarms_def, alarms_adapted, J_eff_series, nu_series


def far_from_alarms(alarms: np.ndarray) -> float:
    alarms = np.asarray(alarms).astype(float).ravel()
    return float(np.mean(alarms))