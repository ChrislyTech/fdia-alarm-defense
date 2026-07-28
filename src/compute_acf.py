import numpy as np

runs_dir = "runs"
idv_list = [2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21]

acf_spe_lag1 = []
acf_spe_max = []

for idv in idv_list:
    try:
        data = np.load(f"{runs_dir}/B_vacf_idv{idv}/series.npz")
        spe = data["SPE_base"]
        n = len(spe)
        spe_c = spe - np.mean(spe)
        var = np.var(spe)
        if var > 0:
            acf = [float(np.mean(spe_c[:n-lag] * spe_c[lag:]) / var)
                   for lag in range(1, 11)]
            acf_spe_lag1.append(acf[0])
            acf_spe_max.append(max(abs(x) for x in acf[:5]))
            print(f"IDV {idv:2d}: ACF lags 1-5 = {[round(x,3) for x in acf[:5]]}")
    except Exception as e:
        print(f"IDV {idv}: {e}")

print(f"\nACF lag-1 — médiane: {np.median(acf_spe_lag1):.3f}, max: {np.max(acf_spe_lag1):.3f}")
print(f"ACF max(lags 1-5) — médiane: {np.median(acf_spe_max):.3f}, max: {np.max(acf_spe_max):.3f}")
