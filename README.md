# PGA-DRL: Progressive Graph Attention-Based Deep Reinforcement Learning

Welcome to the official repository for **PGA-DRL**, a novel model integrating Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs) within a Deep Reinforcement Learning framework to enhance recommender systems. This repository provides the code implementation and supporting materials for the research paper:

**"PGA-DRL: Progressive Graph Attention-Based Deep Reinforcement Learning for Recommender Systems"**

## Features
- Progressive fusion of GCN and GAT embeddings for enhanced global and local user-item interaction representation.
- Actor-Critic framework for optimizing long-term user satisfaction.
- Scalable and efficient implementation suitable for large-scale datasets.
- Evaluation on multiple benchmark datasets with state-of-the-art results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RS-Research/PGA-DRL.git
   cd PGA-DRL
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Python 3.8+ and PyTorch installed.

## Usage

1. Preprocess the dataset:
   ```bash
   python preprocess.py --dataset <dataset_name>
   ```

2. Train the model:
   ```bash
   python train.py --config configs/<config_file>.yaml
   ```

3. Evaluate the model:
   ```bash
   python evaluate.py --checkpoint <path_to_checkpoint>
   ```

## Datasets
The following datasets were used for evaluation:
- ML-100k
- ML-1M
- Amazon Subscription Boxes
- Amazon Magazine Subscriptions
- ModCloth

Preprocessed datasets are available in the `data/` directory or can be generated using the provided scripts.

## Results
PGA-DRL achieves state-of-the-art performance across multiple metrics, including:
- **Precision@10**
- **Recall@10**
- **NDCG@10**
- **MRR@10**
- **Hit@10**

For detailed results, refer to the `results/` directory or the research paper.

## Contributing
We welcome contributions! Please follow these steps:
1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/<feature_name>
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add <feature_name>"
   ```
4. Push to your fork and create a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
This research was conducted by:
- Sang-Woong Lee
- Jawad Tanveer
- Amir Masoud Rahmani
- Khursheed Aurangzeb
- Mahfooz Alam
- Gholamreza Zare
- Pegah Malekpour Alamdari
- Mehdi Hosseinzadeh

For inquiries, please contact: [ghrzarea@gmail.com](mailto:ghrzarea@gmail.com)

---
Thank you for your interest in PGA-DRL! If you use this code in your research, please cite our paper.
## PGA-DRL: Progressive Graph Attention-Based Deep Reinforcement Learning for Recommender Systems

