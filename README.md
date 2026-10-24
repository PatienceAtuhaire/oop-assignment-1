# OOP Mini-Projects

This repository contains two small, reusable Python projects with Jupyter
notebooks and unit tests. The population examples for Kyegegwa and Mubende and
the micro-grid demand data are illustrative synthetic examples, not official
UBOS statistics or measured energy demand.

## Setup and run

Install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Open `project1_population.ipynb` and `project2_microgrid.ipynb` in Jupyter and
run all cells from top to bottom. Reusable modules are in `src/`. Run the unit
tests from the root with:

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

## AI-use declaration

An AI coding assistant was used to help implement the reusable Python modules,
tests and notebook structure. The author should review, run, understand and
adapt all code and analysis before submission.
