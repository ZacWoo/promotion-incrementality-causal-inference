# Data

Place `train.csv` and `store.csv` here (`data/raw/`).

**Source:** [Rossmann Store Sales on Kaggle](https://www.kaggle.com/datasets/shahpranshu27/rossman-store-sales) (same files as the classic competition).

From the project root, after activating the virtual environment:

```bash
python scripts/download_data.py
```

The script uses [Kaggle Hub](https://github.com/Kaggle/kagglehub) and does not require the separate Kaggle CLI.

Do not commit the CSVs if you prefer to keep the repo small; the download script reproduces them.
