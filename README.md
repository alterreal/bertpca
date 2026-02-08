# BertPCa: Transformer-based Dynamic Survival Analysis for Prostate Cancer Post-Surgical Outcomes

BertPCa is a dynamic survival analysis model that uses transformer architectures and Weibull distributions, designed to predict prostate cancer post-surgical biochemical recurrence and cancer-specific mortality. Despite this being the main target during development, BertPCa is quite flexible and can be used with other time-varying time-to-event datasets, as demonstrated here with the PBC2 dataset


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/alterreal/bertpca
   cd bertpca
   ```

2. Install dependencies:

   ```bash
   pip install uv
   uv lock            
   uv sync         
   ```

## Quick Start


### Training

Train with the default configuration (paths and options in `config/config.yaml`):

```bash
uv run python scripts/train_bertpca.py
```

Save the best model to a specific path:

```bash
uv run python scripts/train_bertpca.py --output outputs/models/my_model.keras
```

### Hyperparameter tuning

Run Optuna optimization (default number of trials from config):

```bash
uv run python scripts/tune_bertpca.py
```

Customize trials and study name:

```bash
uv run python scripts/tune_bertpca.py --n-trials 100 --study-name my_study
```

Results and optimization log are written to `outputs/results/`.


## Model Architecture

1. **Input**: Static features (repeated across time) and dynamic features arranged in padded sequences of length `seq_length`.
2. **Transformer encoder**: Multi-head self-attention layers for sequence modeling.
3. **Convolutional blocks**: 1D convolutions with pooling for local feature extraction.
4. **Dense stack**: Fully connected layers.
5. **Weibull head**: Two outputs (alpha, beta) parameterizing the Weibull survival distribution.

## Evaluation Metrics

- **Time-dependent C-index**: Discrimination at prediction times `p_times` and evaluation times `e_times`.
- **Weighted C-index**: `weighted_c_index` in `bertpca.metrics` (inverse probability of censoring weighting).


## License

This project is licensed under the **MIT License**

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.

## Acknowledgments

- Contents of `metrics.py` are credited to Changee Lee
