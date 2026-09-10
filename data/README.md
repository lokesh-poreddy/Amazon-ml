# Data Directory

**NEVER commit actual competition data to Git.**

## Directory Structure

```
data/
├── raw/          ← Place original unmodified competition files here
├── interim/      ← Cleaned / partially processed data
└── processed/    ← Final feature matrices ready for training
```

## Day 1: Place Competition Data

After downloading the competition files:

```bash
cp ~/Downloads/train.csv data/raw/train.csv
cp ~/Downloads/test.csv  data/raw/test.csv
cp ~/Downloads/*.parquet data/raw/   # if applicable
```

Then run:

```bash
jupyter notebook notebooks/00_schema_audit.ipynb
```

This will automatically discover and audit everything in `data/raw/`.
