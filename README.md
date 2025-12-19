# Vector to Vector ID

```
📦 Project Root
├── data/                      # Dataset storage directory
├── saved_models/              # Trained model weights
│
├── .gitignore                 # Git ignore configuration
├── dataset.py                 # Dataset loading and preprocessing script
├── evaluate.py                # Evaluation script
├── hierarchical.py            # Hierarchical clustering implementation
├── model.py                   # Model definition
├── README.md                  # Project description and instructions
├── requirements.txt           # Python dependencies
├── train.py                   # Model training script
├── trie.py                    # Prefix tree implementation
└── utils.py                   # Utility functions
```

## Setup
* check you cuda version via `nvidia-smi`, below command is 2.8.0+cu129.
* python version is 3.11.13.
```
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
```

## Run Training

### Using Product Quantization (PQ) - Default
* Please first reproduce the recall with the default config (at `./saved_models`).
    * For 10K, R@1: 0.387, R@10: 0.741, R@20: 0.808
    * For 100K, R@1: 0.443, R@10: 0.682, R@20: 0.724
* To change the configuration, edit the `get_args()` function in `utils.py`.

```
python train.py --num_samples 10K --quantizer pq
```

### Using Hierarchical Clustering
Hierarchical clustering creates a tree structure where vector IDs represent paths through the hierarchy.
This can potentially capture semantic relationships better than independent PQ subspaces.

```
python train.py --num_samples 10K --quantizer hierarchical
```

## Run Evaluation

### PQ Model
```
python evaluate.py --num_samples 10K --noise_factor 0.0 --quantizer pq
```

### Hierarchical Clustering Model
```
python evaluate.py --num_samples 10K --noise_factor 0.0 --quantizer hierarchical
```

## Quantization Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **PQ** | Splits vectors into M subspaces, clusters each independently | Fast, proven effective | No cross-subspace dependencies |
| **Hierarchical** | Creates a tree of K-means clusters | Captures global structure | More computation during fitting |
