# Data for PGA-DRL

This repository leverages datasets listed on the [RecBole Dataset List](https://recbole.io/dataset_list.html) for training, validation, and evaluation of the **PGA-DRL** model. The datasets are diverse and cater to a wide range of domains in recommender system research.

## Supported Datasets

The following datasets from RecBole have been integrated into this repository:

1. **Movielens**:
   - `ml-100k`: 100,000 user-movie interactions.
   - `ml-1m`: 1,000,000 user-movie interactions.

2. **Amazon**:
   - `Amazon-Subscription-Boxes`: Interactions related to subscription products.
   - `Amazon-Magazine-Subscriptions`: Sparse interactions in the magazine domain.

3. **ModCloth**:
   - Dataset containing user-item interactions from ModCloth’s clothing retail platform.

4. **Yelp**:
   - User reviews and interactions for business recommendations.

## Using the Datasets

1. **Download the Datasets**
   Datasets can be manually downloaded from the [RecBole Dataset List](https://recbole.io/dataset_list.html) or automatically fetched using RecBole.

2. **Setup for PGA-DRL**
   Place the downloaded dataset files in the `data/` directory of this repository. Ensure the following folder structure:

   ```plaintext
   data/
   ├── ml-100k/
   │   ├── ratings.csv
   │   ├── users.csv
   │   └── items.csv
   ├── amazon-subscription-boxes/
   │   ├── interactions.csv
   │   └── metadata.csv
   └── ...
   ```

3. **Configuration**
   Update the `config.yaml` file to specify the dataset you want to use:

   ```yaml
   dataset: ml-100k
   data_path: ./data/ml-100k
   ```

4. **Preprocessing**
   Preprocess the datasets to match the input requirements of the PGA-DRL model:

   ```bash
   python preprocess.py --dataset ml-100k
   ```

## Citation

If you use any datasets from RecBole in your research, please cite their work:

```bibtex
@inproceedings{recbole,
  title={RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
  author={Zhao, Wayne Xin and others},
  booktitle={Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2020}
}
```

## Notes
- Ensure datasets are used in accordance with their respective licenses.
- Use preprocessed datasets to speed up training and evaluation.
