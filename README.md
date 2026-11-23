# Machine-Learning Analysis of Photophobic ALPs at Tera-Z FCC-ee and CEPC

Analysis-code snapshot accompanying **Probing Photophobic Axion-Like Particles via Prompt Leptons, Displaced Vertices, and Mono-γ at Tera-Z FCC-ee and CEPC**.
The study considers e+e− → Z → γa at √s = 91.2 GeV. XGBoost is used for prompt leptonic and mono-γ searches; the displaced-vertex analysis uses dedicated Delphes-LLP reconstruction. Luminosity benchmarks in the manuscript are 25 and 150 ab−1.

This directory follows the workflow-oriented documentation of the [neutrino code repository](https://github.com/NeutrinoMLcode/Machine-learning-code-for-Neutrino). It is a standalone ALP folder that may be added alongside `neutrino/` and `LR4/`, or used in a separate repository. The original event CSVs and ROOT samples are omitted. Scientific source cells have been preserved, including the version differences described in [reproducibility notes](docs/REPRODUCIBILITY.md). This snapshot has been statically checked; a complete paper reproduction has not been run.

## Repository structure

```text
ALP_GitHub_upload/
├── ALP_mu/                   # direct a → μμ
├── ALP_ME/                   # mono-γ (invisible/escaping ALP)
├── ALP_tata/
│   ├── ALP_e/                # a → ττ → ee + neutrinos
│   ├── ALP_mu/               # a → ττ → μμ + neutrinos
│   └── ALP_emu/              # a → ττ → eμ + neutrinos
├── delphes/                  # detector card, LLP scripts and summary tables
├── ALP_scan_result_DV/       # supplied displaced-vertex scan tables
├── responce_2_depend_data/   # supplied updated ML/normalization products
├── Sensitivity/              # supplementary calculation snapshots
├── test3(2)/           # supplementary hadronic-width/contour snapshots
```

Each ML channel has the same structure:

```text
CHANNEL/
├── read_preselection_and_write_plot_significance.ipynb
├── plot.ipynb
├── fit.ipynb
├── pre_read_preselection_and_write.ipynb
├── prediect.ipynb
├── clf_ALP_*.gz[.partNN]    # compressed original models; large archive split into parts
├── train_data/
│   └── select_data/
└── predict_data/
    └── select_data/
```

`prediect.ipynb` retains its original spelling. `ALP_mu/plot.ipynb` is copied from `plot-Copy1.ipynb`. Empty data directories contain `.gitkeep` so their layout survives a Git upload. Notebook outputs and execution counts have been cleared.

## Software requirements

Use Python 3 and a Jupyter Python kernel. Original exact dependency versions are not locked in the research directory, so `requirements.txt` is an unpinned dependency list, not a validated environment lockfile.

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS instead:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python restore_models.py
python -m jupyterlab
```

Some exploratory classifier/search cells set `device='cuda'`. Use an appropriate GPU environment or change this to `cpu` for those cells. The separately constructed final estimators do not all inherit that setting. Delphes/MadGraph/Pythia generation requires its own external toolchain; Python requirements alone do not install it.

## Input data

Supply the original event-level CSVs under the appropriate channel's `train_data/` and `predict_data/`. Do not interchange training and prediction samples.

| Channel | Signal filenames in each raw-data directory | Background |
|---|---|---|
| Direct μμ | `a10.csv`, `a30.csv`, `a50.csv`, `a70.csv`, `a85.csv` | `amumu.csv` |
| Each ττ final state | `a10.csv`, `a30.csv`, `a50.csv`, `a70.csv`, `a85.csv` | `atata.csv` |
| Mono-γ | `1MeV.csv`, `10MeV.csv`, `100MeV.csv`, `1GeV.csv`, `10GeV.csv` | `aME.csv` |

The leptonic signal benchmarks have masses 10, 30, 50, 70 and 85 GeV. The mono-γ benchmarks are 1, 10, 100 MeV and 1, 10 GeV. [data_manifest.json](docs/data_manifest.json) records the actual local CSV filenames, original byte sizes and column headers, including processed samples. Full event samples are not distributed in this package. They must be restored from the research dataset or separately regenerated; no public download endpoint is assumed.

The preprocessing derives photon energy and angular observables and, for dileptons, four-momentum, invariant-mass and ΔR variables. Historical `mu` column labels in the τ→ee implementation are retained; consult the actual Electron input columns instead of inferring particle flavor from derived-column names.

## Execution order

Run each notebook from its own channel directory, from top to bottom. For example, start Jupyter from `ALP_mu/` for the direct dimuon workflow.

1. **Prepare training samples:** run `read_preselection_and_write_plot_significance.ipynb`. It reads `train_data/*.csv` and writes `train_data/select_data/*.csv`.
2. **Plot kinematics:** run `plot.ipynb` using the processed training samples.
3. **Train:** run `fit.ipynb`. For direct μμ this includes an optional, expensive grid search before the fixed final estimator. See the mono-γ exception below.
4. **Prepare prediction samples:** run `pre_read_preselection_and_write.ipynb`. It reads `predict_data/*.csv` and writes `predict_data/select_data/*.csv`. Skip if the original processed samples have already been restored.
5. **Predict and scan thresholds:** run `prediect.ipynb`. It loads the channel's saved classifier, evaluates ROC/score distributions, efficiencies and threshold-dependent CLs, and writes the results CSV and figures.

The original code often writes `results.csv` in the channel directory even when it creates `ML_output/`. Existing result CSVs are supplied snapshots, not fresh validation runs.

## Classifier inputs and settings

| Channel | Input count | Final depth | Trees | Test fraction | Model used for prediction |
|---|---:|---:|---:|---:|---|
| Direct μμ | 9 | 18 | 300 | 0.20 | `clf_ALP_18_300` |
| ττ → ee | 6 | 12 | 300 | 0.50 | `clf_ALP_12_300` |
| ττ → μμ | 6 | 12 | 300 | 0.50 | `clf_ALP_12_300` |
| ττ → eμ | 6 | 12 | 300 | 0.50 | `clf_ALP_12_300` |
| Mono-γ, historical fit | 3 | 6 | 300 | 0.30 | writes `clf_ALP_6_300` |
| Mono-γ, supplied prediction | 2 | not inferred from filename | supplied model | — | `clf_ALP_6_300_2` |

Final training constructors use learning rate 0.02 and `binary:logistic`. Splits use `random_state=42`, but the preceding full-data shuffle has no fixed seed. Exact ordered feature names from the saved-model headers appear in [model_manifest.json](docs/model_manifest.json).

The direct μμ features are `Photon_PT_0`, `photon_energy`, `Muon_PT_0`, `lead_muon_energy`, `Muon_PT_1`, `nextlead_muon_energy`, `deltar_mu_0_mu_1`, `deltar_mu_1_gamma`, `dimuon_mass`.

The ττ→ee and μμ inputs are `Photon_PT_0`, `photon_energy`, `MissingET_MET_0`, `deltar_mu_0_mu_1`, `deltar_mu_0_gamma`, `deltar_mu_1_gamma`. The eμ inputs, in order, are `photon_energy`, `Photon_PT_0`, `deltar_mu_0_e_0`, `deltar_mu_0_gamma`, `deltar_e_0_gamma`, `MissingET_MET_0`.

**Mono-γ version difference:** the manuscript and prediction notebook use `photon_energy` and `photon_theta_cos` (= |cos θγ|). The supplied historical `fit.ipynb` additionally uses `MissingET_MET_0` and writes a different model. Retraining it does not regenerate the two-feature model used by `prediect.ipynb`. Use the supplied `_2` model to inspect the existing prediction workflow; resolving its training provenance requires author review. Both originals are preserved.

## Statistical normalization

Prediction notebooks convert cross sections from pb to fb using 1000 and set L = 150,000 fb−1 = 150 ab−1. The direct μμ production-cross-section list is `[48.436, 35.622, 17.185, 3.4839, 0.10672, 87]`; the ττ list has the same five signal entries and background `84.3`. Signal rates are further multiplied by the ALP branching ratios coded in each notebook. Tau factors are `0.178²` for ee, `0.174²` for μμ, and `0.178*0.174` for eμ, as implemented. The eμ factor's treatment of both charge/flavor assignments needs confirmation before interpreting it as an inclusive rate.

Mono-γ uses `[50.21, 50.21, 50.209, 50.191, 48.42, 14]`, scales the signal by `1/1000²`, and multiplies by escape probabilities `[1, 1, 1, 0.999986755, 0.047942845]` for L > 450 cm. These are benchmark-specific constants, not a general lifetime calculation.

The updated prediction notebooks implement one-channel exact Poisson CLs without nuisance parameters, with pseudo-observation `floor(B+0.5)`, and scan 100 score thresholds. This threshold study is distinct from a validated joint-channel likelihood implementation for the paper's combined exclusion contours. See the limitations below before reproducing numerical limits.

## Displaced vertices and supplementary calculations

`delphes/delphes_card.dat`, ROOT-extraction scripts, model archives and supplied summary tables are included. Some scripts contain an original Linux `BASE_DIR` which must be configured for the user's event location; others expose `--root-dir` / `--out-dir`. Inspect the script before running it. The included 1–200 cm truth/LLP-window helpers are not by themselves proof of the manuscript's complete reconstructed-DV selection.

`Sensitivity/` and `test3(2)/test3/` preserve alternative width, scan and contour calculations with their local supporting tables. Their filenames do not identify an authoritative final version. They are supplementary research snapshots, not a claimed end-to-end reproduction of the manuscript. In `test3(2)/test3/combine_new.ipynb`, the external constraint CSVs are under `limit/`; the original plotting cell refers to bare filenames and needs that path configured.

## Reproducibility, provenance and data availability

Read [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for concrete caveats and [VALIDATION.md](docs/VALIDATION.md) for checks actually run. [source_manifest.json](docs/source_manifest.json) records source-to-package mappings and SHA-256 checksums. Models are compressed to keep their stored GitHub files manageable; `restore_models.py` restores their exact original bytes and verifies their checksums. Restored models and full event data are ignored by Git.

Third-party simulation/model archives retain their own licensing/provenance. No new blanket license has been assigned to them. When adding this folder to the existing repository, review how its existing license applies to the authors' code and the bundled third-party resources.

The direct dimuon model archive is split into ordered 20 MiB `.gz.partNN` files so every packaged file is below the browser upload limit. Keep all parts together; `restore_models.py` joins them automatically. Upload files in batches when the website limits the number per upload. See [GitHub file-size documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
