# Datasets

No data is included in this repository. Download the public CSV releases and place them as follows
(the search is recursive, so unzipped folders work):

| Dataset | File(s) | Place under | Source |
|---|---|---|---|
| Edge-IIoTset | `ML-EdgeIIoT-dataset.csv` | `data/raw/edge_iiotset/` | https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot |
| ToN-IoT | `train_test_network.csv` (Train_Test_Network) | `data/raw/ton_iot/` | https://research.unsw.edu.au/projects/toniot-datasets |
| UNSW-NB15 | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | `data/raw/unsw_nb15/` | https://research.unsw.edu.au/projects/unsw-nb15-dataset |

Then run

```bash
python -m src.data.prepare_datasets --raw-dir data/raw --out-dir data
```

which writes `data/<name>_processed.npz` and records every preprocessing choice and the SHA-256 of each raw file in the
`meta` field. Please cite the dataset authors (see the paper's references) when you use these datasets.
