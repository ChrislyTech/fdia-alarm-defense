# make_figs.py
# ============
# Génération des figures "propres" pour la Section IV à partir
# des artefacts déjà produits par main.py (series.npz, etc.)

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import config as opt

# -------------------------------------------------------------------
# Utils basiques
# -------------------------------------------------------------------

RUNS_DIR = Path("runs")
FIGS_DIR = Path("figs")
FIGS_DIR.mkdir(parents=True, exist_ok=True)


def load_thresholds():
    """
    Charge les seuils J_T2 et J_SPE sauvegardés par main.py
    (runs/thresholds.npz).
    """
    th_path = RUNS_DIR / "thresholds.npz"
    if not th_path.exists():
        raise FileNotFoundError(
            f"{th_path} not found. Run main.py once so thresholds are saved."
        )
    data = np.load(th_path)
    J_T2 = float(data["J_T2"])
    J_SPE = float(data["J_SPE"])
    return J_T2, J_SPE


def load_series(idv: int, tag: str = "B"):
    """
    Charge series.npz pour un IDV donné et un tag de dossier :
      tag = "A" -> runs/A_vacf_idv{idv}
      tag = "B" -> runs/B_vacf_idv{idv}
    """
    series_path = RUNS_DIR / f"{tag}_vacf_idv{idv}" / "series.npz"
    if not series_path.exists():
        raise FileNotFoundError(f"Missing series.npz for IDV={idv}, tag='{tag}' at {series_path}")
    return np.load(series_path)


# -------------------------------------------------------------------
# FIGURES 1–2 : SPE trajectories for Xt9 / Xt4 (pre-fault)
# -------------------------------------------------------------------

def make_xt_spe_fig(idv: int, tag: str, out_path: Path, ylim: tuple = None):
    """
    Produit une figure SPE(base) vs SPE(attack) pour un IDV donné.
    Utilise uniquement series.npz (déjà généré par main.py).
    ylim: optional (ymin, ymax) to enforce a shared y-axis scale across IDVs.
          Spikes exceeding ylim are annotated with a downward arrow to indicate clipping.
    """
    _, J_SPE = load_thresholds()
    data = load_series(idv, tag)

    spe_base = data["SPE_base"]
    spe_attack = data["SPE_attack"]

    n = len(spe_base)
    t = np.arange(n)

    # Index de cut (fin de la fenêtre pre-fault) = opt.f
    cut_idx = int(opt.f)
    cut_idx = min(cut_idx, n - 1)

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(t, spe_base, label="SPE (base)")
    ax.plot(t, spe_attack, label="SPE (attack)")
    ax.axhline(J_SPE, linestyle="--", label=r"$J_{\mathrm{SPE}}$")
    ax.axvline(cut_idx, linestyle=":", color="k", label="window end")
    ax.set_xlim(0, n)
    if ylim is not None:
        ax.set_ylim(ylim)
        # Annotate clipped spikes (attack SPE exceeding ylim)
        ymax = ylim[1]
        clipped = np.where(spe_attack > ymax)[0]
        if len(clipped) > 0:
            # Group consecutive clipped indices, annotate once per group
            groups = np.split(clipped, np.where(np.diff(clipped) > 3)[0] + 1)
            for grp in groups:
                peak_t = grp[np.argmax(spe_attack[grp])]
                ax.annotate(
                    f"↑ {spe_attack[peak_t]:.2f}",
                    xy=(peak_t, ymax),
                    xytext=(peak_t + 2, ymax * 0.92),
                    fontsize=7, color="darkorange",
                    arrowprops=dict(arrowstyle="-", color="darkorange", lw=0.8)
                )
    ax.set_xlabel("t (samples)")
    ax.set_ylabel("SPE")
    ax.set_title(f"Xt{idv}: SPE traces")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[make_figs] SPE figure saved: {out_path}")


def make_xt_spe_all():
    """
    Génère :
      figs/xt12_spe.png  (IDV=12)  [vulnérable / gros lift]
      figs/xt4_spe.png   (IDV=4)   [robuste / faible lift]
    en utilisant les runs tag='B'.
    Supervisor request: IDV 4 y-axis is limited to the same range as IDV 12
    so the two figures are comparable (same scale).
    """
    # Step 1 — compute ylim from IDV 12 data
    _, J_SPE = load_thresholds()
    data12 = load_series(12, "B")
    spe_ref = np.concatenate([data12["SPE_base"], data12["SPE_attack"]])
    shared_ymax = max(np.max(spe_ref), J_SPE) * 1.15   # 15% headroom

    # Step 2 — generate both figures with the shared ylim
    make_xt_spe_fig(12, tag="B", out_path=FIGS_DIR / "xt12_spe.png",
                    ylim=(0, shared_ymax))
    make_xt_spe_fig(4,  tag="B", out_path=FIGS_DIR / "xt4_spe.png",
                    ylim=(0, shared_ymax))


# -------------------------------------------------------------------
# FIGURES 3–4 : gate(t) + alpha(t) pour Xt9 / Xt4
# -------------------------------------------------------------------

def make_xt_gate_alpha_fig(idv: int, tag: str, out_path: Path):
    case_dir = RUNS_DIR / f"{tag}_vacf_idv{idv}"
    series_path = case_dir / "series.npz"
    if not series_path.exists():
        raise FileNotFoundError(f"series.npz not found: {series_path}")

    data = np.load(series_path)

    SPE_base   = data["SPE_base"]
    SPE_attack = data["SPE_attack"]
    _, J_SPE   = load_thresholds()

    T     = len(SPE_base)
    L_on  = 30
    L_off = 15
    L     = L_on + L_off
    rho   = L_on / L
    K     = opt.k if hasattr(opt, 'k') else 3
    S_size = len(data["S"]) if "S" in data.files else 3
    c     = 1e-6

    # --- Reconstruire burst(t) ---
    burst = np.array([(t % L) < L_on for t in range(T)], dtype=float)

    # --- Reconstruire frag_rho(t) depuis SPE_base ---
    q_frag = np.quantile(SPE_base, 1 - rho)
    frag   = (SPE_base >= q_frag).astype(float)

    # --- gate(t) = max(burst, frag) ---
    gate_reconstructed = np.maximum(burst, frag)

    # --- Recalculer alpha_t sample par sample via Eq.(12)-(13) ---
    eps_max = float(np.max(np.abs(data["delta"]))) if "delta" in data.files else 0.15
    alpha_reconstructed = np.zeros(T)
    for t_idx in range(1, T):
        if gate_reconstructed[t_idx] == 0:
            alpha_reconstructed[t_idx] = 0.0
        else:
            d_prev = max(J_SPE - SPE_attack[t_idx - 1], 0.0)
            g_raw  = (K / (S_size + c)) * d_prev
            alpha_reconstructed[t_idx] = min(eps_max, g_raw)

    t_arr    = np.arange(T)
    cut_idx  = int(opt.f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.0), sharex=True)

    ax1.step(t_arr, gate_reconstructed, where="post")
    ax1.set_ylabel("gate(t)")
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_title(f"IDV {idv}: temporal gating signal gate($t$)")
    ax1.axvline(cut_idx, linestyle=":", color="k")

    ax2.plot(t_arr, alpha_reconstructed)
    ax2.set_ylabel(r"$\alpha_t$")
    ax2.set_xlabel("t (samples)")
    ax2.set_title(r"Adaptive attack amplitude $\alpha_t$")
    ax2.axvline(cut_idx, linestyle=":", color="k")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_figs] gate+alpha figure saved: {out_path}")

def make_acf_fig(out_path: Path = FIGS_DIR / "acf_spe_alarm.png"):
    """
    ACF figure — médiane + IQR sur 19 IDVs + IDV 4 en overlay.
    """
    idv_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21]
    max_lag  = 10
    lags     = np.arange(1, max_lag + 1)

    acf_spe_all = []
    acf_alm_all = []

    for idv in idv_list:
        try:
            data   = load_series(idv, "B")
            spe    = data["SPE_base"].astype(float)
            alarms = data["alarms_attack"].astype(float)
            n      = len(spe)

            spe_c = spe - np.mean(spe)
            var_s = np.var(spe)
            acf_s = [float(np.mean(spe_c[:n-lag] * spe_c[lag:]) / var_s)
                     for lag in range(1, max_lag+1)]

            a_c   = alarms - np.mean(alarms)
            var_a = np.var(alarms)
            acf_a = [float(np.mean(a_c[:n-lag] * a_c[lag:]) / var_a)
                     if var_a > 0 else 0.0
                     for lag in range(1, max_lag+1)]

            acf_spe_all.append(acf_s)
            acf_alm_all.append(acf_a)
        except Exception:
            continue

    acf_spe_all = np.array(acf_spe_all)
    acf_alm_all = np.array(acf_alm_all)

    # Stats
    spe_med = np.median(acf_spe_all, axis=0)
    spe_q25 = np.percentile(acf_spe_all, 25, axis=0)
    spe_q75 = np.percentile(acf_spe_all, 75, axis=0)

    alm_med = np.median(acf_alm_all, axis=0)
    alm_q25 = np.percentile(acf_alm_all, 25, axis=0)
    alm_q75 = np.percentile(acf_alm_all, 75, axis=0)

    # IDV 4 individually
    idx4    = idv_list.index(4)
    acf4_s  = acf_spe_all[idx4]
    acf4_a  = acf_alm_all[idx4]

    # 95% CI
    n_approx = 160
    ci = 1.96 / np.sqrt(n_approx)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    for ax, med, q25, q75, acf4, title, ylabel in [
        (axes[0], spe_med, spe_q25, spe_q75, acf4_s,
         "ACF of baseline SPE sequence", "ACF"),
        (axes[1], alm_med, alm_q25, alm_q75, acf4_a,
         "ACF of alarm sequence under attack", "ACF"),
    ]:
        # IQR band
        ax.fill_between(lags, q25, q75, alpha=0.25, color="steelblue",
                        label="IQR (19 IDVs)")
        # Median
        ax.plot(lags, med, color="steelblue", marker="o", markersize=4,
                linewidth=1.5, label="Median (19 IDVs)")
        # IDV 4
        ax.plot(lags, acf4, color="darkorange", marker="s", markersize=4,
                linewidth=1.2, linestyle="--", label="IDV 4")
        # 95% CI
        ax.axhline( ci, color="gray", linestyle=":", linewidth=0.9,
                   label="95% CI")
        ax.axhline(-ci, color="gray", linestyle=":", linewidth=0.9)
        ax.axhline(0,   color="black", linewidth=0.8)

        ax.set_xlabel("Lag")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(lags)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        axes[0].set_ylim(-0.08, 0.17)
        axes[1].set_ylim(-0.08, 0.20)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[make_figs] ACF figure saved: {out_path}")

def make_xt_gate_alpha_all():
    """
    Génère :
      figs/xt12_gate_alpha.png
      figs/xt4_gate_alpha.png
    """
    make_xt_gate_alpha_fig(12, tag="A", out_path=FIGS_DIR / "xt12_gate_alpha.png")
    make_xt_gate_alpha_fig(4,  tag="A", out_path=FIGS_DIR / "xt4_gate_alpha.png")


# -------------------------------------------------------------------
# FIGURES 5–6 : heatmap de delta sur capteurs S (Xt9 / Xt4)
# -------------------------------------------------------------------

def _find_delta_matrix(data: np.lib.npyio.NpzFile):
    # Ajoute quelques alias (selon ton exp_harness)
    for key in ["delta_matrix", "delta_mat", "Delta", "delta_S", "deltaS", "delta"]:
        if key in data:
            return data[key]
    raise KeyError(
        "No delta matrix found in series.npz. "
        "Check the saved key name (exp_harness) and update make_figs.py."
    )


def make_xt_delta_heatmap(idv: int, tag: str, out_path: Path):
    data = load_series(idv, tag)
    delta_mat = np.asarray(_find_delta_matrix(data))
    S_indices = _find_sensor_indices(data)

    # We want mat shaped as (|S|, T) for imshow (rows=sensors, cols=time)
    if delta_mat.ndim != 2:
        raise ValueError(f"delta matrix must be 2D, got shape={delta_mat.shape}")

    # Typical case: (T, |S|) with T>>|S| => transpose
    if delta_mat.shape[0] > delta_mat.shape[1]:
        mat = delta_mat.T
    else:
        mat = delta_mat

    n_sensors, n_time = mat.shape
    cut_idx = int(opt.f)

    plt.figure(figsize=(6.4, 3.0))
    im = plt.imshow(
        mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="bwr",
    )
    plt.colorbar(im, label=r"$\Delta$ (z-score)")
    plt.axvline(cut_idx, linestyle=":", color="k")
    plt.xlim(0, n_time)  # avoids the -0.5..N-0.5 look

    plt.xlabel("t (samples)")
    if S_indices is not None and len(S_indices) == n_sensors:
        plt.yticks(np.arange(n_sensors), [f"sensor {int(s)}" for s in S_indices])
    else:
        plt.ylabel("Selected sensors")

    plt.title(f"Xt{idv}: injected $\\Delta$ on selected sensors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[make_figs] delta heatmap saved: {out_path}")


def make_xt_delta_heatmap_all():
    """
    Génère :
      figs/xt12_delta_heatmap.png
      figs/xt4_delta_heatmap.png
    """
    make_xt_delta_heatmap(12, tag="B", out_path=FIGS_DIR / "xt12_delta_heatmap.png")
    make_xt_delta_heatmap(4,  tag="B", out_path=FIGS_DIR / "xt4_delta_heatmap.png")


def _find_sensor_indices(data: np.lib.npyio.NpzFile):
    """
    Essaie de trouver les indices des capteurs S dans series.npz.
    """
    for key in ["S_indices", "idx_S", "S"]:
        if key in data:
            return data[key]
    # Si non dispo, on ne met simplement pas les labels de capteurs.
    return None


# -------------------------------------------------------------------
# FIGURES 7–9 : FAR lift on fault-free windows (bar, hist, scatter)
# -------------------------------------------------------------------

def compute_far_lifts(idv_list, tag: str = "B"):
    """
    Calcule FAR(base), FAR(attack), ΔFAR pour chaque IDV
    à partir des alarms déjà enregistrées dans series.npz.

    On utilise uniquement une fenêtre fault-free [0 .. f-1] avec f=opt.f.
    """
    f_win = int(opt.f)
    base_list = []
    attack_list = []
    delta_list = []
    labels = []

    for idv in idv_list:
        data = load_series(idv, tag)
        alarms_base = data["alarms_base"]
        alarms_attack = data["alarms_attack"]

        if len(alarms_base) < f_win or len(alarms_attack) < f_win:
            raise ValueError(
                f"Not enough samples for IDV={idv} to compute FAR on fault-free window"
            )

        far_base = float(np.mean(alarms_base[:f_win]))
        far_attack = float(np.mean(alarms_attack[:f_win]))
        delta = far_attack - far_base

        labels.append(f"Xt{idv}")
        base_list.append(far_base)
        attack_list.append(far_attack)
        delta_list.append(delta)

    return labels, np.array(base_list), np.array(attack_list), np.array(delta_list)


def make_far_aggregate_figs():
    """
    Génère :
      figs/far_lift_bar.png
      figs/far_lift_hist.png
      figs/far_base_vs_attack_scatter.png
    pour tous les IDV disponibles dans les runs B_vacf_idv* (FAR target 20 %).
    """
    # IDV list – même que dans main.py pour PHASE 4B
    idv_list = [2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    labels, base, attack, delta = compute_far_lifts(idv_list, tag="B")

    # --- Bar chart de ΔFAR par IDV (fault-free window) ---
    order = np.argsort(delta)[::-1]  # tri décroissant
    labels_ord = [labels[i] for i in order]
    delta_ord = delta[order]

    plt.figure(figsize=(7.0, 3.0))
    plt.barh(labels_ord, delta_ord)
    plt.xlabel(r"$\Delta FAR$ (attack - base)")
    plt.title("Change in fault-free window FAR by fault scenario")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "far_lift_bar.png", dpi=300)
    plt.close()
    print(f"[make_figs] ΔFAR_pre bar chart saved: {FIGS_DIR / 'far_lift_bar.png'}")

    # --- Histogramme de ΔFAR (fault-free window) ---
    plt.figure(figsize=(4.0, 3.0))

    # Fixed bin edges (reproducible)
    dmin = float(np.min(delta))
    dmax = float(np.max(delta))
    bins = np.linspace(dmin, dmax, 9)  # 8 bins with fixed edges
    plt.hist(delta, bins=bins, edgecolor="black")

    # Rug plot (each fault as a tick on the x-axis)

    plt.xlabel(r"$\Delta FAR$ (fault-free window)")
    plt.ylabel("Count")
    plt.title(r"Distribution of $\Delta FAR$ across faults")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "far_lift_hist.png", dpi=300)
    plt.close()
    print(f"[make_figs] ΔFAR_pre histogram saved: {FIGS_DIR / 'far_lift_hist.png'}")

    # --- Scatter FAR(base) vs FAR(attack) on fault-free window ---
    plt.figure(figsize=(4.0, 3.0))
    plt.scatter(base, attack)
    min_val = min(float(base.min()), float(attack.min()))
    max_val = max(float(base.max()), float(attack.max()))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel(r"$FAR$ (base)")
    plt.ylabel(r"$FAR$ (attack)")
    plt.title("Fault-free window FAR: base vs attack")
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "far_base_vs_attack_scatter.png", dpi=300)
    plt.close()
    print(f"[make_figs] FAR_pre scatter saved: {FIGS_DIR / 'far_base_vs_attack_scatter.png'}")

from exp_harness import apply_countermeasure_block, far_from_alarms


def expC_defense_summary(tag: str = "B", out_dir_figs: str = "figs", runs_dir: str = "runs"):
    """
    Computes defended FAR on the SAME pre-fault streams as Experiment A case studies,
    by applying (Kc, nu_max, delta) to stored SPE streams in runs/{tag}_vacf_idv*/series.npz.

    Produces:
      - figs/expC_defense_summary_{tag}.csv
      - figs/expC_defended_far_{tag}.png  (bar/line comparison)
      - figs/expC_attack_vs_defended_scatter_{tag}.png
      - figs/expC_idv12_attack_vs_defended_{tag}.png (illustrative)
    """
    thr = np.load(os.path.join(runs_dir, "thresholds.npz"))
    J = float(thr["J_SPE"])
    sigma = float(thr["sigma_SPE_X0"]) if "sigma_SPE_X0" in thr else None
    if sigma is None:
        raise RuntimeError("sigma_SPE_X0 not found in runs/thresholds.npz. Re-run Phase 2 threshold saving patch.")

    # Countermeasure defaults (match Table in paper)
    Kc = 2
    L = 20
    nu_max = 6
    delta_factor = 0.5

    idvs = list(range(1, 22))
    rows = []

    for idv in idvs:
        d = os.path.join(runs_dir, f"{tag}_vacf_idv{idv}")
        f_series = os.path.join(d, "series.npz")
        if not os.path.exists(f_series):
            # skip silently (in case some idvs were not generated)
            continue

        z = np.load(f_series)
        spe_b = z["SPE_base"]
        spe_a = z["SPE_attack"]
        a_b = z["alarms_base"]
        a_a = z["alarms_attack"]

        # defended streams (apply block on SPE)
        def_b, raw_b, Jb, nub = apply_countermeasure_block(
            spe_b, J, sigma, Kc=Kc, L=L, nu_max=nu_max, delta_factor=delta_factor
        )
        def_a, raw_a, Ja, nua = apply_countermeasure_block(
            spe_a, J, sigma, Kc=Kc, L=L, nu_max=nu_max, delta_factor=delta_factor
        )

        far_base = far_from_alarms(a_b)
        far_attack = far_from_alarms(a_a)
        far_def_base = far_from_alarms(def_b)
        far_def_attack = far_from_alarms(def_a)

        rows.append([idv, far_base, far_attack, far_def_base, far_def_attack])

    if len(rows) == 0:
        raise RuntimeError(f"No series.npz found under runs/{tag}_vacf_idv*/. Did you run the IDV case studies?")

    rows = np.array(rows, dtype=float)
    rows = rows[np.argsort(rows[:, 0])]

    # Save CSV
    os.makedirs(out_dir_figs, exist_ok=True)
    csv_path = os.path.join(out_dir_figs, f"expC_defense_summary_{tag}.csv")
    header = "idv,far_base,far_attack,far_def_base,far_def_attack"
    np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
    print(f"[make_figs] ExpC defense CSV saved: {csv_path}")

    # --- Figure 1: defended vs attack FAR per fault (attack & defended only) ---
    idv = rows[:, 0].astype(int)
    far_attack = rows[:, 2]
    far_def_attack = rows[:, 4]

    plt.figure(figsize=(10, 3.2))
    x = np.arange(len(idv))
    w = 0.38
    plt.bar(x - w/2, far_attack, width=w, label="Attack FAR (raw)")
    plt.bar(x + w/2, far_def_attack, width=w, label="Defended FAR (after CM)")
    plt.xticks(x, idv.astype(int))
    plt.xlabel("Fault IDV")
    plt.ylabel("FAR (fault-free window)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best")
    out1 = os.path.join(out_dir_figs, f"expC_defended_far_{tag}.png")
    plt.tight_layout()
    plt.savefig(out1, dpi=300)
    plt.close()
    print(f"[make_figs] ExpC defended FAR figure saved: {out1}")

    # --- Figure 2: scatter attack FAR vs defended FAR ---
    plt.figure(figsize=(4.8, 4.2))
    plt.scatter(far_attack, far_def_attack)
    m = max(float(np.max(far_attack)), float(np.max(far_def_attack)))
    plt.plot([0, m], [0, m], linestyle="--", linewidth=1.2)
    plt.xlabel("Attack FAR (raw)")
    plt.ylabel("Defended FAR")
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(out_dir_figs, f"expC_attack_vs_defended_scatter_{tag}.png")
    plt.tight_layout()
    plt.savefig(out2, dpi=300)
    plt.close()
    print(f"[make_figs] ExpC scatter figure saved: {out2}")

    # --- Figure 3: Fig. 7 for paper — IDV 6 selected after systematic analysis ---
    # IDV 6 is the best candidate: J_eff(t) is clearly time-varying (stepwise),
    # toggle-rate exceeds nu_max multiple times, K_c suppression is visible.
    for target_idv in [6]:
        d_idv = os.path.join(runs_dir, f"{tag}_vacf_idv{target_idv}", "series.npz")
        if not os.path.exists(d_idv):
            print(f"[make_figs] series.npz not found for IDV={target_idv}, skipping.")
            continue

        z = np.load(d_idv)
        spe_a = z["SPE_attack"]

        def_a, raw_a, J_eff, nu = apply_countermeasure_block(
            spe_a, J, sigma, Kc=Kc, L=L, nu_max=nu_max, delta_factor=delta_factor
        )

        t = np.arange(len(spe_a))

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(10, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [2.5, 1, 1]}
        )

        # ── Subplot 1: SPE + thresholds ──
        ax1.plot(t, spe_a, linewidth=1.0, color="steelblue", label="SPE (attack)")
        ax1.axhline(J, linestyle="--", linewidth=1.2, color="orange",
                    label=r"$J_{\mathrm{SPE}}$ (fixed)")
        ax1.plot(t, J_eff, linestyle="-.", linewidth=1.4, color="green",
                 label=r"$J_{\mathrm{eff}}(t)$ (adaptive, time-varying)")
        ax1.set_ylabel("SPE value")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Experiment C — IDV {target_idv}: countermeasure block decomposition")

        # ── Subplot 2: toggle-rate nu_t ──
        ax2.plot(t, nu, linewidth=1.0, color="purple", label=r"Toggle-rate $\nu_t$")
        ax2.axhline(nu_max, linestyle="--", linewidth=1.0, color="red",
                    label=r"$\nu_{\max}$" + f" = {nu_max}")
        ax2.set_ylabel(r"$\nu_t$")
        ax2.grid(True, alpha=0.3)
        flutter = (nu > nu_max).astype(float)
        update_active = flutter > 0
        ax2.fill_between(t, 0, nu_max, where=update_active, alpha=0.20, color="red",
                         label=r"Threshold update applied ($\nu_t > \nu_{\max}$)")
        ax2.legend(loc="upper right", fontsize=8)

        # ── Subplot 3: alarm after adaptive threshold vs final defended alarm ──
        ax3.step(t, raw_a, where="post", linewidth=1.0, color="tomato",
                 label=r"Alarm after adaptive threshold: $a_t = \mathbf{1}\{\mathrm{SPE}(z_t) > J_{\mathrm{eff}}(t)\}$")
        ax3.step(t, def_a, where="post", linewidth=1.4, color="red",
                 label=r"Final alarm $a'_t$ after consecutive-$K_c$" + f" ($K_c$={Kc})")
        ax3.set_ylabel("Alarm")
        ax3.set_xlabel("t (samples)")
        ax3.set_ylim(-0.05, 1.05)   # clearly binary
        ax3.set_yticks([0, 1])
        ax3.legend(loc="upper right", fontsize=8)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        out_idv = os.path.join(out_dir_figs, f"expC_idv{target_idv}_attack_vs_defended_{tag}.png")
        fig.savefig(out_idv, dpi=300)
        plt.close(fig)
        print(f"[make_figs] ExpC IDV{target_idv} figure saved: {out_idv}")
        print(
            f"[make_figs] Suggested caption for IDV {target_idv}:\n"
            f"  'Experiment C — IDV {target_idv}: countermeasure block decomposition. "
            f"Top: attacked SPE and thresholds J_SPE (fixed) and J_eff(t) (adaptive). "
            f"Middle: toggle-rate nu_t with limit nu_max={nu_max}; shaded intervals indicate nu_t > nu_max "
            f"where the stepwise threshold update is triggered (with the implementation's one-step timing). "
            f"Bottom: alarm stream after adaptive thresholding a_t = 1{{SPE(z_t) > J_eff(t)}} "
            f"and final persisted alarm a'_t after the consecutive-K_c policy (K_c={Kc}).'"
        )

    # Keep the original IDV12 filename for backward compatibility with paper
    import shutil
    src = os.path.join(out_dir_figs, f"expC_idv12_attack_vs_defended_{tag}.png")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir_figs, f"expC_idv12_attack_vs_defended_{tag}_backup.png"))

# -------------------------------------------------------------------
# Main pour ce script
# -------------------------------------------------------------------

def main():
    # Vérification rapide
    if not RUNS_DIR.exists():
        raise FileNotFoundError("The 'runs' directory does not exist. Run main.py first.")

    # Générer toutes les figures nécessaires pour la Section IV
    make_xt_spe_all()
    make_xt_gate_alpha_all()
    make_xt_delta_heatmap_all()
    make_far_aggregate_figs()
    make_acf_fig()

    expC_defense_summary(tag="B")


if __name__ == "__main__":
    main()

#if __name__ == "__main__":
#    make_acf_fig()