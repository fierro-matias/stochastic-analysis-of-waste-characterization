import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.stats import gaussian_kde, norm

# ============================================================
# 1) Read + extract data
# ============================================================
def read_raw_csv(path, sep=";", decimal="."):
    # robust for ° and Windows Excel exports
    return pd.read_csv(path, sep=sep, decimal=decimal, encoding="cp1252")


def extract_series_with_labels(df, value_col, label_col="Subtype"):
    """Extract numeric values and aligned labels, dropping NaNs in the value column."""
    vals = np.asarray(df[value_col].to_numpy(), dtype=float)
    labels = df[label_col].to_numpy()
    m = ma.masked_invalid(vals)
    labels = ma.masked_array(labels, mask=ma.getmask(m)).compressed()
    vals = m.compressed()
    return labels, vals


def extract_all_properties(df,op):
    """
    Returns a dict with labels/values for each property needed in this script.
    Columns must exist in your CSV.
    """
    if op==0:
        out = {}
        out["rho_L"], out["rho_V"] = extract_series_with_labels(df, "Density g/cm^3")
        out["k_L"],   out["k_V"]   = extract_series_with_labels(df, "Thermal conductivity W/mK")
        out["cp_L"],  out["cp_V"]  = extract_series_with_labels(df, "Specific Heat Capacity kJ/kg*K")
        out["hhv_L"], out["hhv_V"] = extract_series_with_labels(df, "E_HHV [MJ/kg] DAF")
    else:
        out = {}
        out["rho_L"], out["rho_V"] = extract_series_with_labels(df, "Density g/cm^3")
        out["k_L"],   out["k_V"]   = extract_series_with_labels(df, "Thermal conductivity W/mK")
        out["cp_L"],  out["cp_V"]  = extract_series_with_labels(df, "Specific Heat Capacity kJ/kg*K")
    return out


# ============================================================
# 2) Grouping helper 
# ============================================================
def counts_per_subtype(labels):
    """
    Count datapoints per subgroup label in the ORDER OF APPEARANCE of unique labels.
    This matches your original approach where values are already contiguous per subgroup.
    """
    u = np.unique(labels)
    dp = np.array([(labels == ui).sum() for ui in u], dtype=int)
    return dp, u


def split_by_counts(values, dp):
    """Split a 1D values array into a list of groups using dp counts."""
    cuts = np.cumsum(dp)
    cuts = np.concatenate([[0], cuts])
    return [values[cuts[i]:cuts[i+1]] for i in range(len(dp))]


# ============================================================
# 3) PDF models per subgroup and joint (mean of subgroup PDFs)
# ============================================================
def kde_generator_sub(samples, x_vals):
    kde = gaussian_kde(samples)
    return kde(x_vals)


def pdf_collector(groups, x_grid, nn_kde=9):
    
    """
    For each subgroup:
      nn in [3..8]  -> Gaussian fit
      nn >= 9       -> KDE
    Returns list of pdf arrays (each same size as x_grid).
    """
    
    pdfs = []
    types = []  # 0=Gaussian, 1=KDE
    
    for g in groups:
        
        nn = len(g)
        
        if 3 <= nn <= nn_kde-1:
            
            mu, sigma = norm.fit(g)
            
            # guard against too-narrow sigma
            
            if mu != 0 and (sigma / abs(mu)) < 0.03:
                sigma = abs(mu) * 0.03
                
            pdfs.append(norm.pdf(x_grid, loc=mu, scale=sigma))
            types.append(0)
            
        elif nn >= nn_kde:
            
            pdfs.append(kde_generator_sub(g, x_grid))
            types.append(1)
            
    return np.asarray(pdfs), np.asarray(types)


def pdf_joint(pdfs, x_grid):
    """Joint PDF = mean of subgroup PDFs + normalized. Returns None if no valid subgroups."""
    if pdfs.size == 0:
        return None
    pj = np.mean(pdfs, axis=0)
    area = np.trapz(pj, x_grid)
    if area <= 0:
        return None
    return pj / area


# ============================================================
# 4) Moments / quantiles from PDF
# ============================================================
def moments_from_pdf(x, pdf):
    mu = np.trapz(x * pdf, x)
    var = np.trapz((x - mu) ** 2 * pdf, x)
    return mu, np.sqrt(max(var, 0.0))


def quantile_from_pdf(x, pdf, q):
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * np.diff(x))
    cdf = np.concatenate([[0.0], cdf])
    return np.interp(q, cdf, x)


def print_pdf_stats(tag, x, pdf):
    mu, sig = moments_from_pdf(x, pdf)
    q16 = quantile_from_pdf(x, pdf, 0.16)
    q50 = quantile_from_pdf(x, pdf, 0.50)
    q84 = quantile_from_pdf(x, pdf, 0.84)
    print(f"{tag}: mean={mu:.4f}  median={q50:.4f}  q16={q16:.4f}  q84={q84:.4f}  std={sig:.4f}")


def print_sample_stats(tag, samples):
    mean = samples.mean()
    med  = np.median(samples)
    q16  = np.quantile(samples, 0.16)
    q84  = np.quantile(samples, 0.84)
    print(f"{tag}: mean={mean:.4f}  median={med:.4f}  q16={q16:.4f}  q84={q84:.4f}")


# ============================================================
# 5) Monte Carlo mixture samplers
# ============================================================
def build_cdfs(x, pdfs):
    """Normalize each PDF and build its CDF on grid x."""
    x = np.asarray(x)
    pdfs = np.asarray(pdfs)
    dx = np.diff(x)
    dx = np.concatenate(([dx[0]], dx))

    cdfs = np.zeros_like(pdfs)
    for i in range(pdfs.shape[0]):
        pdf = pdfs[i]
        cdf = np.cumsum(pdf * dx)
        cdf[-1] = 1.0
        cdfs[i] = cdf
    return cdfs

def MC_PDFs(x, pdfs, weights, n_samples, n_elements, iti):
    x = np.asarray(x)
    pdfs = np.asarray(pdfs)
    weights = np.asarray(weights)

    rng = np.random.default_rng()

    assert pdfs.shape[0] == n_elements
    assert np.isclose(weights.sum(), 1.)

    dx = np.diff(x)
    dx = np.concatenate(([dx[0]], dx))

    cdfs = np.zeros_like(pdfs)
    for i in range(n_elements):
        pdf = pdfs[i]
        cdf = np.cumsum(pdf * dx)
        cdf[-1] = 1.0
        cdfs[i] = cdf

    u = rng.random((n_elements, n_samples))

    if iti == 0:
        denom = np.zeros(n_samples)
        for i in range(n_elements):
            xi = np.interp(u[i], cdfs[i], x)
            denom += weights[i] / xi
        return 1.0 / denom

    elif iti == 2:
        y = np.zeros(n_samples)
        for i in range(n_elements):
            xi = np.interp(u[i], cdfs[i], x)
            y += weights[i] * xi
        return y

    return np.zeros(n_samples)


def MC_PDFs_ks(x_rho, x_ks, pdfs_rho, pdfs_ks, weights, n_samples, n_elements):
    x_rho = np.asarray(x_rho)
    x_ks = np.asarray(x_ks)
    pdfs_rho = np.asarray(pdfs_rho)
    pdfs_ks = np.asarray(pdfs_ks)
    weights = np.asarray(weights)

    rng = np.random.default_rng()

    assert pdfs_rho.shape[0] == n_elements
    assert pdfs_ks.shape[0] == n_elements
    assert np.isclose(weights.sum(), 1.)

    dx_rho = np.diff(x_rho)
    dx_rho = np.concatenate(([dx_rho[0]], dx_rho))

    cdfs_rho = np.zeros_like(pdfs_rho)
    for i in range(n_elements):
        cdf = np.cumsum(pdfs_rho[i] * dx_rho)
        cdf[-1] = 1.0
        cdfs_rho[i] = cdf

    u_rho = rng.random((n_elements, n_samples))

    rho_i = np.zeros((n_elements, n_samples))
    denom = np.zeros(n_samples)
    for i in range(n_elements):
        rho_i[i] = np.interp(u_rho[i], cdfs_rho[i], x_rho)
        denom += weights[i] / rho_i[i]

    x_vol = np.zeros_like(rho_i)
    for i in range(n_elements):
        x_vol[i] = (weights[i] / rho_i[i]) / denom

    dx_ks = np.diff(x_ks)
    dx_ks = np.concatenate(([dx_ks[0]], dx_ks))

    cdfs_ks = np.zeros_like(pdfs_ks)
    for i in range(n_elements):
        cdf = np.cumsum(pdfs_ks[i] * dx_ks)
        cdf[-1] = 1.0
        cdfs_ks[i] = cdf

    u_ks = rng.random((n_elements, n_samples))
    k_mix = np.ones(n_samples)
    for i in range(n_elements):
        ki = np.interp(u_ks[i], cdfs_ks[i], x_ks)
        k_mix *= ki ** x_vol[i]

    return x_vol, k_mix

def kde_pdf_from_samples(samples, n_grid):
    """
    Build a KDE PDF from samples and return (x_vals, pdf_vals).
    Normalizes PDF numerically.
    """
    samples = np.asarray(samples, dtype=float)
    kde = gaussian_kde(samples)
    x_vals = np.linspace(samples.min(), samples.max(), n_grid)
    pdf_vals = kde(x_vals)

    area = np.trapz(pdf_vals, x_vals)
    if area > 0:
        pdf_vals /= area
    return x_vals, pdf_vals

# ============================================================
# 6) Main runner
# ============================================================
def main():
    # --- config ---
    seed = None  # set None for non-reproducible runs

    n_grid = 20000
    n_mc   = 1_000_000

    # common grids
    x_rho = np.linspace(0.0001, 6.0, n_grid)
    x_k   = np.linspace(0.0, 2.5, n_grid)
    x_cp  = np.linspace(0.0, 5.5, n_grid)
    x_hhv = np.linspace(0.0, 55.0, n_grid)

    # waste cases (Paper, Organic, Plastic)
    w_cases_3 = np.array([ [16.6, 39.9, 25.9],
                           [41.2, 41.2,  0.0],
                           [41.2,  0.0, 41.2],
                           [ 0.0, 41.2, 41.2],
                         ], dtype=float)
    
    w_cases_3 /= w_cases_3.sum(axis=1, keepdims=True)

    # same cases but with Inert added as constant mass fraction 17.6%

    w_cases_4 = np.array([ [16.6, 39.9, 25.9, 17.6],
                           [41.2, 41.2,  0.0, 17.6],
                           [41.2,  0.0, 41.2, 17.6],
                           [ 0.0, 41.2, 41.2, 17.6],
                         ], dtype=float)
    
    w_cases_4 /= w_cases_4.sum(axis=1, keepdims=True)

    print(w_cases_4)

    # --- load + extract ---
    df_P = read_raw_csv("../data/raw/paper_subgroups_raw_data.csv")
    df_O = read_raw_csv("../data/raw/organic_subgroups_raw_data.csv")
    df_S = read_raw_csv("../data/raw/plastic_subgroups_raw_data.csv")
    df_I = read_raw_csv("../data/raw/inert_subgroups_raw_data.csv")

    P = extract_all_properties(df_P,0)
    O = extract_all_properties(df_O,0)
    S = extract_all_properties(df_S,0)
    I = extract_all_properties(df_I,1)

    # --- build subgroup counts and groups (values assumed contiguous by subtype) ---
    P_rho_dp, _ = counts_per_subtype(P["rho_L"])
    O_rho_dp, _ = counts_per_subtype(O["rho_L"])
    S_rho_dp, _ = counts_per_subtype(S["rho_L"])
    I_rho_dp, _ = counts_per_subtype(I["rho_L"])

    P_k_dp, _ = counts_per_subtype(P["k_L"])
    O_k_dp, _ = counts_per_subtype(O["k_L"])
    S_k_dp, _ = counts_per_subtype(S["k_L"])
    I_k_dp, _ = counts_per_subtype(I["k_L"])

    P_cp_dp, _ = counts_per_subtype(P["cp_L"])
    O_cp_dp, _ = counts_per_subtype(O["cp_L"])
    S_cp_dp, _ = counts_per_subtype(S["cp_L"])
    I_cp_dp, _ = counts_per_subtype(I["cp_L"])

    P_hhv_dp, _ = counts_per_subtype(P["hhv_L"])
    O_hhv_dp, _ = counts_per_subtype(O["hhv_L"])
    S_hhv_dp, _ = counts_per_subtype(S["hhv_L"])

    P_rho_groups = split_by_counts(P["rho_V"], P_rho_dp)
    O_rho_groups = split_by_counts(O["rho_V"], O_rho_dp)
    S_rho_groups = split_by_counts(S["rho_V"], S_rho_dp)
    I_rho_groups = split_by_counts(I["rho_V"], I_rho_dp)

    # keep your special filter
    O_rho_groups = [g[g > 0.25] for g in O_rho_groups]

    P_k_groups = split_by_counts(P["k_V"], P_k_dp)
    O_k_groups = split_by_counts(O["k_V"], O_k_dp)
    S_k_groups = split_by_counts(S["k_V"], S_k_dp)
    I_k_groups = split_by_counts(I["k_V"], I_k_dp)

    P_cp_groups = split_by_counts(P["cp_V"], P_cp_dp)
    O_cp_groups = split_by_counts(O["cp_V"], O_cp_dp)
    S_cp_groups = split_by_counts(S["cp_V"], S_cp_dp)
    I_cp_groups = split_by_counts(I["cp_V"], I_cp_dp)

    P_hhv_groups = split_by_counts(P["hhv_V"], P_hhv_dp)
    O_hhv_groups = split_by_counts(O["hhv_V"], O_hhv_dp)
    S_hhv_groups = split_by_counts(S["hhv_V"], S_hhv_dp)

    # --- build main-group PDFs for rho, k, cp (Paper/Organic/Plastic/Inert) ---
    
    def build_joint_for_property(groups_dict, x_grid, prop_name):
        """
        groups_dict keys: P,O,S,I -> list of subgroup arrays
        returns: pdf_joint for each main group in fixed order [P,O,S,I]
        """
        pdf_joint_4 = np.zeros((4, x_grid.size), dtype=float)
        for idx, key in enumerate(["P", "O", "S", "I"]):
            pdfs, types = pdf_collector(groups_dict[key], x_grid)
            pj = pdf_joint(pdfs, x_grid)
            pdf_joint_4[idx] = pj
            # info print
            print_pdf_stats(f"{prop_name} main {key}", x_grid, pj)
        print()
        return pdf_joint_4

    rho_joint_4 = build_joint_for_property({"P": P_rho_groups, "O": O_rho_groups, "S": S_rho_groups, "I": I_rho_groups}, x_rho, "rho")
    k_joint_4 = build_joint_for_property({"P": P_k_groups, "O": O_k_groups, "S": S_k_groups, "I": I_k_groups}, x_k, "k")
    cp_joint_4 = build_joint_for_property({"P": P_cp_groups, "O": O_cp_groups, "S": S_cp_groups, "I": I_cp_groups}, x_cp, "cp")

    # --- HHV_DAF main-group PDFs (Paper/Organic/Plastic only) ---
    def build_joint_hhv(groups_list, x_grid, label):
        pdfs, _ = pdf_collector(groups_list, x_grid)
        pj = pdf_joint(pdfs, x_grid)
        print_pdf_stats(f"hhv main {label}", x_grid, pj)
        return pj

    hhv_joint_3 = np.zeros((3, x_hhv.size), dtype=float)
    hhv_joint_3[0] = build_joint_hhv(P_hhv_groups, x_hhv, "P")
    hhv_joint_3[1] = build_joint_hhv(O_hhv_groups, x_hhv, "O")
    hhv_joint_3[2] = build_joint_hhv(S_hhv_groups, x_hhv, "S")
    print()

    # --- save main-group PDFs (same layout idea as your old file) ---
    names_main = ["x_rho", "x_k", "x_cp", "x_hhv",
                  "rho_Paper", "rho_Organic", "rho_Plastic", "rho_Inert",
                  "k_Paper", "k_Organic", "k_Plastic", "k_Inert",
                  "cp_Paper", "cp_Organic", "cp_Plastic", "cp_Inert",
                  "hhv_Paper", "hhv_Organic", "hhv_Plastic"
                 ]

    results_main = np.column_stack([x_rho, x_k, x_cp, x_hhv,
                                    rho_joint_4[0], rho_joint_4[1], rho_joint_4[2], rho_joint_4[3],
                                    k_joint_4[0],   k_joint_4[1],   k_joint_4[2],   k_joint_4[3],
                                    cp_joint_4[0],  cp_joint_4[1],  cp_joint_4[2],  cp_joint_4[3],
                                    hhv_joint_3[0], hhv_joint_3[1], hhv_joint_3[2]
                                   ])
    
    print()
    out_main = f"PDFs_TPP_main_groups_{int(n_mc/1000)}k_cases.csv"
    pd.DataFrame(results_main, columns=names_main).to_csv(out_main, index=False)
    print("Wrote:", out_main, "\n")

    # ========================================================
    # 7) Waste cases  (all MC runs independent: RNG created inside MC_* )
    # ========================================================
    
    # --- HHV waste: only P/O/S, arithmetic mixture (iti=2) ---
    hhv_waste_x   = np.zeros((4, n_grid), dtype=float)
    hhv_waste_pdf = np.zeros((4, n_grid), dtype=float)
    
    for case_i in range(4):
        # independent MC stream inside MC_PDFs
        samples = MC_PDFs(
            x_hhv,
            hhv_joint_3,            # shape (3, n_grid)
            w_cases_3[case_i],      # shape (3,)
            n_samples=n_mc,
            n_elements=3,
            iti=2
        )
        print_sample_stats(f"HHV waste {chr(ord('A') + case_i)}", samples)
    
        x_vals, pdf_vals = kde_pdf_from_samples(samples, n_grid)
        hhv_waste_x[case_i]   = x_vals
        hhv_waste_pdf[case_i] = pdf_vals
    
    print()
    
    # --- rho/k/cp waste ---
    # rho: harmonic mean (iti=0)
    # k: geometric mean with volume fractions from rho
    # cp: arithmetic mean (iti=2)
    tot_waste_x   = np.zeros((3, 4, n_grid), dtype=float)   # properties [rho,k,cp], cases A..D
    tot_waste_pdf = np.zeros((3, 4, n_grid), dtype=float)
    
    # rho
    for case_i in range(4):
        samples = MC_PDFs(
            x_rho,
            rho_joint_4,            # shape (4, n_grid)
            w_cases_4[case_i],      # shape (4,)
            n_samples=n_mc,
            n_elements=4,
            iti=0
        )
        print_sample_stats(f"rho waste {chr(ord('A') + case_i)}", samples)
    
        x_vals, pdf_vals = kde_pdf_from_samples(samples, n_grid)
        tot_waste_x[0, case_i]   = x_vals
        tot_waste_pdf[0, case_i] = pdf_vals
    
    print()
    
    # k (needs rho joint + k joint)
    for case_i in range(4):
        # independent MC stream inside MC_PDFs_ks
        _, k_samples = MC_PDFs_ks(
            x_rho,
            x_k,
            rho_joint_4,            # shape (4, n_grid)
            k_joint_4,              # shape (4, n_grid)
            w_cases_4[case_i],      # shape (4,)
            n_samples=n_mc,
            n_elements=4
        )
        print_sample_stats(f"k waste {chr(ord('A') + case_i)}", k_samples)
    
        x_vals, pdf_vals = kde_pdf_from_samples(k_samples, n_grid)
        tot_waste_x[1, case_i]   = x_vals
        tot_waste_pdf[1, case_i] = pdf_vals
    
    print()
    
    # cp (arithmetic)
    for case_i in range(4):
        samples = MC_PDFs(
            x_cp,
            cp_joint_4,            # shape (4, n_grid)
            w_cases_4[case_i],     # shape (4,)
            n_samples=n_mc,
            n_elements=4,
            iti=2
        )
        print_sample_stats(f"cp waste {chr(ord('A') + case_i)}", samples)
    
        x_vals, pdf_vals = kde_pdf_from_samples(samples, n_grid)
        tot_waste_x[2, case_i]   = x_vals
        tot_waste_pdf[2, case_i] = pdf_vals
    
    print()

    # --- save waste PDFs ---
    names_waste = ["x_rho_A","x_rho_B","x_rho_C","x_rho_D",
                   "x_k_A","x_k_B","x_k_C","x_k_D",
                   "x_cp_A","x_cp_B","x_cp_C","x_cp_D",
                   "x_hhv_A","x_hhv_B","x_hhv_C","x_hhv_D",
                   "rho_A","rho_B","rho_C","rho_D",
                   "k_A","k_B","k_C","k_D",
                   "cp_A","cp_B","cp_C","cp_D",
                   "hhv_A","hhv_B","hhv_C","hhv_D",
                  ]

    results_waste = np.column_stack([tot_waste_x[0,0], tot_waste_x[0,1], tot_waste_x[0,2], tot_waste_x[0,3],
                                     tot_waste_x[1,0], tot_waste_x[1,1], tot_waste_x[1,2], tot_waste_x[1,3],
                                     tot_waste_x[2,0], tot_waste_x[2,1], tot_waste_x[2,2], tot_waste_x[2,3],
                                     hhv_waste_x[0], hhv_waste_x[1], hhv_waste_x[2], hhv_waste_x[3],
                                     tot_waste_pdf[0,0], tot_waste_pdf[0,1], tot_waste_pdf[0,2], tot_waste_pdf[0,3],
                                     tot_waste_pdf[1,0], tot_waste_pdf[1,1], tot_waste_pdf[1,2], tot_waste_pdf[1,3],
                                     tot_waste_pdf[2,0], tot_waste_pdf[2,1], tot_waste_pdf[2,2], tot_waste_pdf[2,3],
                                     hhv_waste_pdf[0], hhv_waste_pdf[1], hhv_waste_pdf[2], hhv_waste_pdf[3],
                                     ])

    out_waste = f"PDFs_TPP_waste_samples_{int(n_mc/1000)}k_cases.csv"
    pd.DataFrame(results_waste, columns=names_waste).to_csv(out_waste, index=False)
    print("Wrote:", out_waste)

if __name__ == "__main__":
    main()