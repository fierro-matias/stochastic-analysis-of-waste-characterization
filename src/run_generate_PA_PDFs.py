import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.stats import gaussian_kde

# ============================================================
# 1) Read + extract proximate analysis data (Moisture Content / Volatile Matter / Char / Ash)
# ============================================================
def read_raw_csv(path, sep=";", decimal="."):
    """Read a raw subgroup table."""
    return pd.read_csv(path, sep=sep, decimal=decimal)  


def extract_numeric_with_labels(df, value_col, label_col="Subtype"):
    """
    Extract values and labels, dropping NaNs in the value column.
    Returns:
      labels: 1D array of subtype ids/names (aligned with values)
      values: 1D float array
    """
    vals = np.asarray(df[value_col].to_numpy(), dtype=float)
    labels = df[label_col].to_numpy()

    m = ma.masked_invalid(vals)
    labels = ma.masked_array(labels, mask=ma.getmask(m)).compressed()
    vals = m.compressed()
    return labels, vals


def extract_PA_inputs(df):
    """
    Extract proximate-analysis components (already normalized columns).
    Returns:
      subtype labels for M_norm, and four aligned value arrays
    """
    L, M = extract_numeric_with_labels(df, "M_norm")
    _, V = extract_numeric_with_labels(df, "V_norm")
    _, C = extract_numeric_with_labels(df, "C_norm")
    _, A = extract_numeric_with_labels(df, "A_norm")
    return L, M, V, C, A

# ============================================================
# 2) ILR transform/inverse (Helmert basis)
# ============================================================
def _helmert_basis(D):
    """
    Build an orthonormal ILR basis (Helmert sub-matrix via QR).
    Returns Q with shape (D, D-1).
    """
    H = np.zeros((D, D - 1))
    for i in range(1, D):
        H[:i, i - 1] = 1.0 / i
        H[i,  i - 1] = -1.0
    Q, _ = np.linalg.qr(H, mode="reduced")  # shape (D, D-1)
    return Q


def ilr_transform(X):
    """
    ILR transform for compositions (rows sum to 1, strictly positive).
    X: (n, D)
    Returns: (n, D-1)
    """
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("ILR requires strictly positive parts (use a small epsilon fix).")

    n, D = X.shape
    Q = _helmert_basis(D)

    logX = np.log(X)
    centered = logX - logX.mean(axis=1, keepdims=True)  # CLR-centered
    return centered @ Q


def ilr_inverse(Y):
    """
    Inverse ILR transform consistent with ilr_transform above.
    Y: (n, D-1) -> X: (n, D), rows sum to 1
    """
    Y = np.asarray(Y, dtype=float)
    n, d = Y.shape
    D = d + 1
    Q = _helmert_basis(D)

    logX = Y @ Q.T
    X = np.exp(logX)
    X /= X.sum(axis=1, keepdims=True)
    return X


# ============================================================
# 3) Normalize PA + generate subgroup samples
# ============================================================
def normalize_PA(M, V, C, A, eps=5e-3):
    """
    Ensure positivity, then normalize to sum=1.
    Inputs can be vectors (same length).
    eps in %
    """
    M = np.maximum(M, eps)
    V = np.maximum(V, eps)
    C = np.maximum(C, eps)
    A = np.maximum(A, eps)
    S = M + V + C + A
    return np.column_stack([M / S, V / S, C / S, A / S])


def split_by_subtype(labels, X):
    """Split rows of X by unique subtype labels, preserving per-subgroup arrays."""
    out = []
    for s in np.unique(labels):
        m = (labels == s)
        out.append(X[m])
    return out


def sample_subgroup_model(X_sub, n_samples, nn_kde=9):
    """
    Fit subgroup in ILR space and resample:
      - small nn (3..8): MVN in ILR (mean + covariance)
      - large nn (>=9): KDE in ILR
    """
    nn = X_sub.shape[0]
    Y = ilr_transform(X_sub)

    if 3 <= nn <= nn_kde-1:
        
        mu = Y.mean(axis=0)
        cov = np.cov(Y, rowvar=False)
        Ys = np.random.multivariate_normal(mu, cov, size=n_samples)

        return ilr_inverse(Ys)
        
    elif nn >= nn_kde:
        
        kde = gaussian_kde(Y.T)
        Ys = kde.resample(n_samples).T

        return ilr_inverse(Ys)
    
    return np.empty((0, X_sub.shape[1]))

def generate_group_samples(df, n_samples_per_subgroup):
    """
    Main-group pipeline:
      raw df -> PA components -> normalize -> split by subtype -> resample each subtype
    Returns: list of arrays, one per subtype: (n_samples_per_subgroup, 4)
    """
    L, M, V, C, A = extract_PA_inputs(df)
    X = normalize_PA(M, V, C, A)

    subgroups = split_by_subtype(L, X)
    samples_per_subgroup = [ sample_subgroup_model(Xs, n_samples_per_subgroup) for Xs in subgroups ]

    samples_per_subgroup = [ S for S in samples_per_subgroup if S.shape[0] > 0 ]
    
    return samples_per_subgroup


# ============================================================
# 4) Mixtures + boundary-correct KDE PDFs
# ============================================================
def reflection_kde_pdf(x_grid, samples, bw_method=None):
    """
    Boundary-correct KDE on [0,1] using reflection.
    Only for visulization !!
    """
    x = np.asarray(x_grid, dtype=float)
    s = np.asarray(samples, dtype=float)
    s = s[(s >= 0.0) & (s <= 1.0)]
    if s.size < 2:
        return np.zeros_like(x)

    s_ref = np.concatenate([-s, s, 2.0 - s])
    kde = gaussian_kde(s_ref, bw_method=bw_method)
    pdf = kde(x)

    area = np.trapz(pdf, x)
    if area > 0:
        pdf /= area
    return pdf


def mixture_samples(samples_per_group, weights, n_total):
    """
    Draw samples from a mixture of subgroups:
      - choose subgroup index ~ weights
      - resample rows uniformly from that subgroup pool
    """
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    counts = np.random.multinomial(n_total, weights)
    out = []
    for Xg, c in zip(samples_per_group, counts):
        idx = np.random.randint(0, Xg.shape[0], size=c)
        out.append(Xg[idx])
    return np.vstack(out)


def pdfs_for_group(samples_per_subgroup, x_grid, n_total_pdf, weights=None, bw=None):
    """
    Produce 4 marginal PDFs (one per PA component) for a main group.
    Only for visulization !!
    """
    G = len(samples_per_subgroup)
    if weights is None:
        weights = np.ones(G) / G

    X = mixture_samples(samples_per_subgroup, weights, n_total=n_total_pdf)

    pdf = np.zeros((4, x_grid.size), dtype=float)
    for j in range(4):
        pdf[j] = reflection_kde_pdf(x_grid, X[:, j], bw_method=bw)
    return pdf

# ============================================================
# 5) Summary printing
# ============================================================

def print_summary(label, X, symbols):
    """
    Print mean / median / q16 / q84 for an already-built array X (n, D).
    """
    mean_vals = X.mean(axis=0)
    med_vals  = np.median(X, axis=0)
    q16_vals  = np.quantile(X, 0.16, axis=0)
    q84_vals  = np.quantile(X, 0.84, axis=0)

    print(label)
    for k, s in enumerate(symbols):
        print(f"  {s}: mean={mean_vals[k]:.5f}  median={med_vals[k]:.5f}  q16={q16_vals[k]:.5f}  q84={q84_vals[k]:.5f}")
    print(f"  sum(mean)={mean_vals.sum():.6f}")
    print()
    
# ============================================================
# 6) Waste blending
# ============================================================
def draw_waste_blend(samples_by_main, w_main, n_total):
    """
    Draw waste compositions as convex combinations of main-group draws.
    Each main group is itself drawn from equal-weight subgroups.
    """
    w_main = np.asarray(w_main, dtype=float)
    w_main /= w_main.sum()

    X = np.zeros((n_total, 4), dtype=float)

    for mg, (subgroups, wm) in enumerate(zip(samples_by_main, w_main)):
        w_sg = np.ones(len(subgroups)) / len(subgroups)  # equal subgroup weights inside main group
        sg_ids = np.random.choice(len(subgroups), size=n_total, p=w_sg)

        Xm = np.zeros((n_total, 4), dtype=float)
        for sgi in range(len(subgroups)):
            pick = np.where(sg_ids == sgi)[0]
            if pick.size == 0:
                continue
            S = subgroups[sgi]
            rr = np.random.randint(0, S.shape[0], size=pick.size)
            Xm[pick] = S[rr]

        X += wm * Xm

    X /= X.sum(axis=1, keepdims=True)
    return X


def apply_inert_to_ash(X, inert_mass_fraction):
    """
    Scale PA components by (1-inert) and add inert to ash (index 3).
    """
    inert = float(inert_mass_fraction)
    scale = 1.0 - inert
    Y = X * scale
    Y[:, 3] += inert
    Y /= Y.sum(axis=1, keepdims=True)
    return Y

# ============================================================
# 7) Main script
# ============================================================
def main():
    # --- config ---
    n_grid = 20000
    x_grid = np.linspace(0.0, 1.0, n_grid)

    n_samples_per_subgroup = 1_000_000     # subgroup model resampling size
    n_total_pdf_main = 1_000_000           # samples used to estimate each main-group PDF
    n_total_waste = 1_000_000              # samples used to estimate each waste-case PDF

    # Waste cases: rows = scenarios A..D, columns = [Paper, Organic, Plastic]
    weights_cases = np.array([ [16.6, 39.9, 25.9],
                               [41.2, 41.2,  0.0],
                               [41.2,  0.0, 41.2],
                               [ 0.0, 41.2, 41.2],
                             ], dtype=float)
    
    weights_cases /= weights_cases.sum(axis=1, keepdims=True)

    inert = 17.6 / 100.0 # In all waste samples.

    # --- load raw data ---
    df_P = read_raw_csv("../data/raw/paper_subgroups_raw_data.csv")
    df_O = read_raw_csv("../data/raw/organic_subgroups_raw_data.csv")
    df_S = read_raw_csv("../data/raw/plastic_subgroups_raw_data.csv")

    # --- generate subgroup pools for each main group ---
    samples_P = generate_group_samples(df_P, n_samples_per_subgroup)
    samples_O = generate_group_samples(df_O, n_samples_per_subgroup)
    samples_S = generate_group_samples(df_S, n_samples_per_subgroup)
    print("PA Subgroup samples ready.\n")

    
    symbols_PA = ["Water", "Vol_M", "Char", "Ash"]

    print_summary("Main Group: Paper",  np.vstack(samples_P), symbols_PA)
    print_summary("Main Group: Organic", np.vstack(samples_O), symbols_PA)
    print_summary("Main Group: Plastic", np.vstack(samples_S), symbols_PA)

    print()
    # --- main-group PDFs (equal-weight mixture over subgroups) ---
    pdf_P = pdfs_for_group(samples_P, x_grid, n_total_pdf_main)
    print("Paper Main Group done.")
    pdf_O = pdfs_for_group(samples_O, x_grid, n_total_pdf_main)
    print("Organic Main Group done.")
    pdf_S = pdfs_for_group(samples_S, x_grid, n_total_pdf_main)
    print("Plastic Main Group done.")

    # Save main-group PDFs
    names_main = ["x_PA",
                  "Water_Paper","Vol_M_Paper","Char_Paper","Ash_Paper",
                  "Water_Organic","Vol_M_Organic","Char_Organic","Ash_Organic",
                  "Water_Plastic","Vol_M_Plastic","Char_Plastic","Ash_Plastic"
                 ]
    
    results_main = np.column_stack([x_grid,
                                    pdf_P[0], pdf_P[1], pdf_P[2], pdf_P[3],
                                    pdf_O[0], pdf_O[1], pdf_O[2], pdf_O[3],
                                    pdf_S[0], pdf_S[1], pdf_S[2], pdf_S[3]
                                ])
    
    print()
    out_main = f"PDFs_PA_main_groups_{int(n_total_pdf_main/1000)}k_cases.csv"
    pd.DataFrame(results_main, columns=names_main).to_csv(out_main, index=False)
    print("Wrote:", out_main)

    # --- waste PDFs for each case A..D ---
    samples_by_main = [samples_P, samples_O, samples_S]

    pdf_waste = np.zeros((weights_cases.shape[0], 4, x_grid.size), dtype=float)

    for case_i in range(weights_cases.shape[0]):
        Xw = draw_waste_blend(samples_by_main, weights_cases[case_i], n_total=n_total_waste)
        Xw = apply_inert_to_ash(Xw, inert_mass_fraction=inert)
        
        print_summary(f"Waste Case {chr(ord('A') + case_i)}",Xw,symbols_PA)

        for j in range(4):
            pdf_waste[case_i, j] = reflection_kde_pdf(x_grid, Xw[:, j])


    names_waste = ["x_PA",
                   "Water_A","Vol_M_A","Char_A","Ash_A",
                   "Water_B","Vol_M_B","Char_B","Ash_B",
                   "Water_C","Vol_M_C","Char_C","Ash_C",
                   "Water_D","Vol_M_D","Char_D","Ash_D"
                  ]

    results_waste = np.column_stack([x_grid,
                                     pdf_waste[0,0], pdf_waste[0,1], pdf_waste[0,2], pdf_waste[0,3],
                                     pdf_waste[1,0], pdf_waste[1,1], pdf_waste[1,2], pdf_waste[1,3],
                                     pdf_waste[2,0], pdf_waste[2,1], pdf_waste[2,2], pdf_waste[2,3],
                                     pdf_waste[3,0], pdf_waste[3,1], pdf_waste[3,2], pdf_waste[3,3]
                                    ])

    out_waste = f"PDFs_PA_waste_samples_{int(n_total_waste/1000)}k_cases.csv"
    pd.DataFrame(results_waste, columns=names_waste).to_csv(out_waste, index=False)
    print("Wrote:", out_waste)

if __name__ == "__main__":
    main()