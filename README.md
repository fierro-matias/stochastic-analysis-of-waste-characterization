[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18928774.svg)](https://doi.org/10.5281/zenodo.18928774)
# Waste Characterization and Stochastic Analysis

This repository contains a curated database and stochastic characterization of thermophysical and compositional properties of municipal solid waste (MSW). The dataset compiles literature-derived measurements and provides probability density functions (PDFs) for key waste properties obtained through hierarchical statistical aggregation.

The objective of this repository is to provide a reproducible framework for the statistical representation of waste material properties, enabling uncertainty-aware estimation of waste properties for experimental studies and numerical simulations.

---

## Repository contents

The repository includes:

- A structured database of waste characterization data (proximate analysis, elemental composition, and thermophysical properties).
- Stochastic distributions for waste **main groups** (Paper, Organic, Plastic, and Inert).
- Derived distributions for representative **waste mixtures** (Baseline, S1–S3).
- Precomputed probability density functions (PDFs) generated using large-scale Monte Carlo sampling (10⁶ realizations). 
- Python scripts used to generate the distributions and export the resulting datasets.
- Visualization notebooks for inspecting and comparing the resulting PDFs.

---

## Data structure

The dataset contains three main categories of waste properties:

### Proximate Analysis (PA)
- Moisture content  
- Volatile matter  
- Fixed carbon  
- Ash  

### Elemental Analysis (EA)
- Carbon (C)  
- Hydrogen (H)  
- Nitrogen (N)  
- Sulfur (S)  
- Oxygen (O)  
- Chlorine (Cl)

### Thermophysical Properties (TPP)
- Density (ρ)  
- Thermal conductivity (k)  
- Heat capacity (cₚ)  
- Higher heating value (HHV)

For each property, probability density functions are provided for both **main waste groups** and **representative waste mixtures**.

Proximate Analysis (PA) and Elemental Analysis (EA) data are reported as provided in the original sources and subsequently normalized to a common representative basis to ensure comparability.

---

## Repository structure

```
data/
    raw/
        waste_database_master.xlsx
        paper_subgroups_raw_data.csv
        organic_subgroups_raw_data.csv
        plastic_subgroups_raw_data.csv
        inert_subgroups_raw_data.csv
        references_index.csv
  
    pdfs/
        PA/
            PDFs_PA_main_groups_1000k_cases.csv
            PDFs_PA_waste_samples_1000k_cases.csv
        EA/
            PDFs_EA_main_groups_1000k_cases.csv
            PDFs_EA_waste_samples_1000k_cases.csv
        TPP/
            PDFs_TPP_main_groups_1000k_cases.csv
            PDFs_TPP_waste_samples_1000k_cases.csv
src/
    run_generate_TPP_PDFs.py
    run_generate_PA_PDFs.py
    run_generate_EA_PDFs.py

notebooks/
    pdfs_plot.ipynb

```

- **data/raw**: master database compiled from literature together with processed CSV files organized by waste subgroup.
- **data/pdfs**: derived datasets, including precomputed PDFs.
- **src**: scripts used to generate stochastic distributions.
- **notebooks**: visualization and exploration tools.

---

## Usage

The precomputed PDFs can be used directly for statistical analysis or uncertainty propagation in waste conversion models.

Example workflow:

1. Load the PDF datasets from `data/pdfs/`.
2. Use the provided plotting notebook (`notebooks/pdfs_plot.ipynb`) to visualize distributions.
3. Optionally regenerate the PDFs from the raw dataset using the scripts provided in `src/`. 

---

## Extending the database

Researchers may extend the database by adding new literature measurements to the master database (`data/raw/waste_database_master.xlsx`). Updated datasets can then be generated using the provided scripts.

---

## Citation

If you use this dataset in academic work, please cite the associated publication and the repository release (Zenodo DOI will be added after publication).

Please also cite the original literature sources when using individual measurements from the database.

---

## License

Code: see `LICENSE-CODE`.



The dataset in this repository compiles measurements reported in the scientific literature, public technical reports, and manufacturer datasheets.  
The database structure, aggregation methodology, and derived probability distributions provided here are released under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

Users of this dataset must cite both:
1. The original literature sources when using individual measurements.
2. This repository when using the compiled database or derived probability distributions.

---
## Literature Sources and Reference Index

All measurements contained in the database were extracted from previously published scientific literature, technical reports, and reference books.

To ensure transparency and traceability, the repository includes a reference index file:

```
data/raw/references_index.csv
```

This file lists all literature sources used in the compilation of the dataset. Each reference entry includes:

- **ID** – unique identifier used to link database entries to the original source
- **Citation** – full bibliographic citation of the publication
- **DOI** – Digital Object Identifier (when available)
- **Web** – direct link to the publication or document
- **Year** – publication year
- **Country** – country where the measurements were reported or conducted
- **Continent** – geographic region of the study
- **Type of reference** – classification of the source (e.g., scientific publication or technical report)
- **Ref Paper / Ref Organic / Ref Plastic** – indicators showing which waste material groups the reference contributes data for

The reference index allows users to:

1. Trace each measurement back to its original publication.
2. Verify the context and methodology of the reported values.
3. Properly cite the original literature when reusing individual measurements.

Users of this repository should cite both the original literature sources listed in `references_index.csv` and the repository itself when using the compiled dataset or derived probability distributions.

---

## Reproducibility

All probability density functions provided in `data/pdfs/` were generated using Monte Carlo sampling (10⁶ realizations) from the compiled literature database.

The scripts used to generate these distributions are provided in `src/` and allow full regeneration of the datasets.
