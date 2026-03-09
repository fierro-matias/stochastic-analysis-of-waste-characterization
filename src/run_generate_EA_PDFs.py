import numpy as np
import numpy.ma as ma
import pandas as pd
from scipy.stats import gaussian_kde

# ============================================================
# 1) Read + extract elemental analysis data (C/H/N/S/O)
# ============================================================
def read_raw_csv(path, sep=";", decimal="."):
    """
     Read a raw subgroup table.
     """
    return pd.read_csv(path, sep=sep, decimal=decimal)

def extract_numeric_with_labels(df, value_col, label_col="Subtype"):
    """Return labels + values after removing NaNs in the value column."""
    vals = np.asarray(df[value_col].to_numpy(), dtype=float)
    labels = df[label_col].to_numpy()

    m = ma.masked_invalid(vals)
    labels = ma.masked_array(labels, mask=ma.getmask(m)).compressed()
    vals = m.compressed()
    return labels, vals


def extract_EA_inputs(df):
    """
    Extract EA normalized mass fractions (columns must exist).
    Returns:
      labels (Subtype) aligned with C,H,N,S,O arrays
    """
    L, C = extract_numeric_with_labels(df, "E_C_norm")
    _, H = extract_numeric_with_labels(df, "E_H_norm")
    _, N = extract_numeric_with_labels(df, "E_N_norm")
    _, S = extract_numeric_with_labels(df, "E_S_norm")
    _, O = extract_numeric_with_labels(df, "E_O_norm")
    return L, C, H, N, S, O


# ============================================================
# 2) ILR transform/inverse (Helmert basis)
# ============================================================
def _helmert_basis(D):
    """Return orthonormal basis Q with shape (D, D-1)."""
    H = np.zeros((D, D - 1))
    for i in range(1, D):
        H[:i, i - 1] = 1.0 / i
        H[i,  i - 1] = -1.0
    Q, _ = np.linalg.qr(H, mode="reduced")
    return Q


def ilr_transform(X):
    """
    ILR transform:
      X: (n, D) compositions, strictly positive, rows sum to 1
      returns (n, D-1)
    """
    X = np.asarray(X, dtype=float)
    if np.any(X <= 0.0):
        raise ValueError("ILR requires strictly positive parts (use epsilon correction).")
    n, D = X.shape
    Q = _helmert_basis(D)

    logX = np.log(X)
    centered = logX - logX.mean(axis=1, keepdims=True)  # CLR-centered
    return centered @ Q


def ilr_inverse(Y):
    """
    Inverse ILR:
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
# 3) Normalize EA + generate subgroup samples
# ============================================================
def normalize_EA(C, H, N, S, O, eps=5e-3):
    """
    Epsilon-correct and normalize to sum=1.
    eps in %
    """
    C = np.maximum(C, eps)
    H = np.maximum(H, eps)
    N = np.maximum(N, eps)
    S = np.maximum(S, eps)
    O = np.maximum(O, eps)
    tot = C + H + N + S + O
    return np.column_stack([C/tot, H/tot, N/tot, S/tot, O/tot])


def split_by_subtype(labels, X):
    """Split X by unique subtype labels into a list of arrays."""
    out = []
    for s in np.unique(labels):
        out.append(X[labels == s])
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


def generate_group_samples_EA(df, n_samples_per_subgroup):
    """
    EA main-group pipeline:
      raw df -> EA columns -> normalize -> split by subtype -> resample each subtype
    Returns: list of arrays, one per subtype: (n_samples_per_subgroup, 5)
    """
    L, C, H, N, S, O = extract_EA_inputs(df)
    X = normalize_EA(C, H, N, S, O)
    subgroups = split_by_subtype(L, X)

    samples_per_subgroup = [
        sample_subgroup_model(Xs, n_samples_per_subgroup) for Xs in subgroups
    ]

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
    Draw samples from subgroup mixture (with replacement).
    """
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    counts = np.random.multinomial(n_total, weights)
    out = []
    for Xg, c in zip(samples_per_group, counts):
        idx = np.random.randint(0, Xg.shape[0], size=c)
        out.append(Xg[idx])
    return np.vstack(out)


def pdfs_for_group_EA(samples_per_subgroup, x_grid, n_total_pdf, weights=None, bw=None):
    """
    Compute 5 marginal PDFs (C,H,N,S,O) for a main group.
    Only for visulization !!
    """
    G = len(samples_per_subgroup)
    if weights is None:
        weights = np.ones(G) / G

    X = mixture_samples(samples_per_subgroup, weights, n_total=n_total_pdf)

    pdf = np.zeros((5, x_grid.size), dtype=float)
    for j in range(5):
        pdf[j] = reflection_kde_pdf(x_grid, X[:, j], bw_method=bw)
    return pdf, X  # return X as well (useful for printing stats)


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
def draw_waste_blend_EA(samples_by_main, w_main, n_total):
    """
    Draw waste compositions:
      - for each draw, pick a subgroup within each main group (equal weight)
      - sample one composition from that subgroup
      - convex combine using w_main
    """
    w_main = np.asarray(w_main, dtype=float)
    w_main /= w_main.sum()

    X = np.zeros((n_total, 5), dtype=float)

    for subgroups, wm in zip(samples_by_main, w_main):
        w_sg = np.ones(len(subgroups)) / len(subgroups)
        sg_ids = np.random.choice(len(subgroups), size=n_total, p=w_sg)

        Xm = np.zeros((n_total, 5), dtype=float)
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


# ============================================================
# 7) Main script
# ============================================================
def main():
    # --- config ---
    symbols = ["C", "H", "N", "S", "O"]

    n_grid = 20000
    x_grid = np.linspace(0.0, 1.0, n_grid)

    n_samples_per_subgroup = 1_000_000   # subgroup resampling size
    n_total_pdf_main = 1_000_000         # samples used to estimate each main-group pdf + stats
    n_total_waste = 1_000_000            # waste sample size per case

    # Waste cases A..D (Paper, Organic, Plastic)
    weights_cases = np.array([
        [16.6, 39.9, 25.9],
        [41.2, 41.2,  0.0],
        [41.2,  0.0, 41.2],
        [ 0.0, 41.2, 41.2],
    ], dtype=float)
    weights_cases /= weights_cases.sum(axis=1, keepdims=True)

    # --- load ---
    df_P = read_raw_csv("../data/raw/paper_subgroups_raw_data.csv")
    df_O = read_raw_csv("../data/raw/organic_subgroups_raw_data.csv")
    df_S = read_raw_csv("../data/raw/plastic_subgroups_raw_data.csv")

    # --- subgroup pools per main group ---
    samples_P = generate_group_samples_EA(df_P, n_samples_per_subgroup)
    samples_O = generate_group_samples_EA(df_O, n_samples_per_subgroup)
    samples_S = generate_group_samples_EA(df_S, n_samples_per_subgroup)
    print("EA subgroup samples ready.\n")

    # --- main-group prints (stats from subgroup pools, not from pdf mixture draws) ---
    print_summary("Main Group: Paper",   np.vstack(samples_P), symbols)
    print_summary("Main Group: Organic", np.vstack(samples_O), symbols)
    print_summary("Main Group: Plastic", np.vstack(samples_S), symbols)
    
    # --- main-group PDFs (still computed the same way) ---
    pdf_P, _ = pdfs_for_group_EA(samples_P, x_grid, n_total_pdf_main)
    print("Paper Main Group done.")
    pdf_O, _ = pdfs_for_group_EA(samples_O, x_grid, n_total_pdf_main)
    print("Organic Main Group done.")
    pdf_S, _ = pdfs_for_group_EA(samples_S, x_grid, n_total_pdf_main)
    print("Plastic Main Group done.")

    # Save main-group PDFs
    names_main = ["x_EA",
                  "C_Paper","H_Paper","N_Paper","S_Paper","O_Paper",
                  "C_Organic","H_Organic","N_Organic","S_Organic","O_Organic",
                  "C_Plastic","H_Plastic","N_Plastic","S_Plastic","O_Plastic"
                 ]
    
    results_main = np.column_stack([x_grid,
                                    pdf_P[0], pdf_P[1], pdf_P[2], pdf_P[3], pdf_P[4],
                                    pdf_O[0], pdf_O[1], pdf_O[2], pdf_O[3], pdf_O[4],
                                    pdf_S[0], pdf_S[1], pdf_S[2], pdf_S[3], pdf_S[4],
                                   ])

    print()
    out_main = f"PDFs_EA_main_groups_{int(n_total_pdf_main/1000)}k_cases.csv"
    pd.DataFrame(results_main, columns=names_main).to_csv(out_main, index=False)
    print("Wrote:", out_main, "\n")

    # --- waste cases A..D: blend + prints + PDFs ---
    samples_by_main = [samples_P, samples_O, samples_S]
    pdf_waste = np.zeros((weights_cases.shape[0], 5, x_grid.size), dtype=float)

    for case_i in range(weights_cases.shape[0]):
        Xw = draw_waste_blend_EA(samples_by_main, weights_cases[case_i], n_total=n_total_waste)
        print_summary(f"Waste Case {chr(ord('A') + case_i)}", Xw, symbols)

        #for j in range(5):
        #    pdf_waste[case_i, j] = reflection_kde_pdf(x_grid, Xw[:, j])

    # Save waste PDFs
    names_waste = ["x_EA",
                   "C_A","H_A","N_A","S_A","O_A",
                   "C_B","H_B","N_B","S_B","O_B",
                   "C_C","H_C","N_C","S_C","O_C",
                   "C_D","H_D","N_D","S_D","O_D"
                  ]
    
    results_waste = np.column_stack([x_grid,
                                     pdf_waste[0,0], pdf_waste[0,1], pdf_waste[0,2], pdf_waste[0,3], pdf_waste[0,4],
                                     pdf_waste[1,0], pdf_waste[1,1], pdf_waste[1,2], pdf_waste[1,3], pdf_waste[1,4],
                                     pdf_waste[2,0], pdf_waste[2,1], pdf_waste[2,2], pdf_waste[2,3], pdf_waste[2,4],
                                     pdf_waste[3,0], pdf_waste[3,1], pdf_waste[3,2], pdf_waste[3,3], pdf_waste[3,4],
                                    ])
    
    out_waste = f"PDFs_EA_waste_samples_{int(n_total_waste/1000)}k_cases.csv"
    pd.DataFrame(results_waste, columns=names_waste).to_csv(out_waste, index=False)
    print("Wrote:", out_waste)

if __name__ == "__main__":
    main()