# Reproduction

## Analysis

Open `CREADS_reproduction.ipynb` in Google Colab with a CPU runtime. Run the cells in order and upload `data/CREADS_Colab_Repro_Inputs.zip` at S1. The archive is extracted to `/content/creads`; output paths are printed by the notebook.

The input bundle contains the processed 340-region labelled table, the 584-region inventory, analysis modules and archived results. Some cells estimate models from the tables; others summarize or plot the supplied CSV and JSON results. The older `98REGIONS` and `300REGIONS` filenames are compatibility aliases: their table values are the same as the 340-region and 584-region files.

## Figures

- Figure 1: open `figures/CREADS_Figure1.ipynb` and upload `figures/CREADS_Figure1_inputs.zip`. The script downloads GADM boundaries from the provider on its first run. About 915 MB is downloaded; subsequent runs reuse the local files. Use the data under the provider's terms.
- Figure 2: run `python figures/build_fig2_workflow.py`. It writes PNG and PDF files beside the script and requires Matplotlib. No input data are needed.

## Environment and checks

The analysis was checked locally in Python 3.12 with the versions in `requirements.txt`. All 29 code cells completed. Hosted Colab execution has not been separately verified. The pooled ensemble yielded overall R² 0.8797 and within-country R² 0.3312, compared with 0.8798 and 0.3318 in the manuscript. This does not establish agreement for every supplementary result; remaining audit differences are shown in the notebook output.

The distributed notebook corrects variable references, input paths and joins that prevented execution in S6d, S9, S11, S15 and S22. S7b retains a visible warning when the legacy silhouette maximum differs. The analysis modules and numerical input values are unchanged. Dependency versions are fixed in S0; the Figure 1 uploader checks archive contents before selecting a ZIP.
