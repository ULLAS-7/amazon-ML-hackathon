# Amazon ML Challenge 2026 - [Your Team Name]

## Team

- [Name 1] - EDA & Data

- [Name 2] - Preprocessing & Features  

- [Name 3] - Modeling

- [Name 4] - Docs & Strategy

## Folder Structure

- data/raw/ - original dataset (not tracked in git)

- data/processed/ - cleaned/engineered data

- notebooks/eda/ - exploratory analysis, one notebook per person, name as eda_<yourname>.ipynb

- notebooks/baseline/ - baseline model notebooks

- notebooks/experiments/ - later iteration notebooks

- src/ - reusable Python modules (preprocessing.py, features.py, model.py, utils.py)

- submissions/ - generated prediction files, name as submission_<timestamp>.csv

- docs/approach.md - the 1-2 page approach document

## Workflow

- main branch is always working and mergeable

- Each person works on their own branch: eda-<name>, model-<name>, etc.

- Merge into main every 1-2 hours, not at the end

- Commit and push at least every 30-60 minutes

## Setup

1. Clone the repo

2. Create virtual environment: python -m venv venv

3. Activate it and run: pip install -r requirements.txt
