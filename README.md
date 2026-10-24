# OOP Mini-Projects

This repository contains reusable Python mini-projects with Jupyter
notebooks and unit tests. The population examples for Kyegegwa and Mubende and
the micro-grid demand data are illustrative synthetic examples, not official
UBOS statistics or measured energy demand.

## Setup and run

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Open `project1_population.ipynb`, `project2_microgrid.ipynb`,
`project3_fishery.ipynb`, `project4_rainfall.ipynb` and `project5_taxi.ipynb`
in Jupyter and run all cells from top to bottom.
Reusable modules are in `src/`. Run the unit tests from the root with:

```bash
python -m pytest
```

GitHub Actions runs the test suite for pushes and pull requests.

## Project summaries

- **Population forecasting:** compares three forecasting methods against a
  held-out period, then estimates 2025–2029 population and incremental
  classroom needs. Details and limitations are in the population notebook.
- **Micro-grid dispatch:** solves a two-source daily demand system, flags
  infeasible solutions, reports usage and costs, and demonstrates a
  three-source extension. Details and limitations are in the micro-grid
  notebook.
- **Fish-stock and export risk:** compares logistic stock scenarios, simulates
  bounded export-price paths and estimates revenue risk. Parameters and
  results are illustrative; see the fishery notebook for assumptions and
  limitations.
- **Rainfall and crop suitability:** compares illustrative monthly rainfall
  patterns, screens crop rainfall suitability, detects rainy-season peaks and
  fetches NASA POWER historical data for a variability extension.
- **Taxi-route planning:** compares route revenue, evaluates passenger
  forecasters with walk-forward validation, solves a supply-demand equilibrium
  and estimates buffered fleet sizes.

## AI-use declaration

An AI coding assistant was used to help implement the reusable Python modules,
tests and notebook structure. The author should review, run, understand and
adapt all code and analysis before submission.
