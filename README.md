# Anomaly-Transformer (ICLR 2022 Spotlight)
Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy

Unsupervised detection of anomaly points in time series is a challenging problem, which requires the model to learn informative representation and derive a distinguishable criterion. In this paper, we propose the Anomaly Transformer in these three folds:

- An inherent distinguishable criterion as **Association Discrepancy** for detection.
- A new **Anomaly-Attention** mechanism to compute the association discrepancy.
- A **minimax strategy** to amplify the normal-abnormal distinguishability of the association discrepancy.

<p align="center">
<img src=".\pics\structure.png" height = "350" alt="" align=center />
</p>

## Get Started

1. Install Python 3.6, PyTorch >= 1.4.0. 
(Thanks Élise for the contribution in solving the environment. See this [issue](https://github.com/thuml/Anomaly-Transformer/issues/11) for details.)
2. Download data. You can obtain four benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1gisthCoE-RrKJ0j3KPV7xiibhHWT9qRm?usp=sharing). **All the datasets are well pre-processed**. For the SWaT dataset, you can apply for it by following its official tutorial.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results as follows:
```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
```

After training completes, you can evaluate the saved checkpoint with
```bash
python main.py --mode test [your args]
```
Use the same arguments as during training&mdash;especially the `--model_tag`
option&mdash;so that the correct model is loaded for testing.

Especially, we use the adjustment operation proposed by [Xu et al, 2018](https://arxiv.org/pdf/1802.03903.pdf) for model evaluation. If you have questions about this, please see this [issue](https://github.com/thuml/Anomaly-Transformer/issues/14) or email us.

## Main Result

We compare our model with 15 baselines, including THOC, InterFusion, etc. **Generally,  Anomaly-Transformer achieves SOTA.**

<p align="center">
<img src=".\pics\result.png" height = "450" alt="" align=center />
</p>

## Continual Experiment

`incremental_experiment.py` now trains a single model on the full dataset. When
CPD (change-point detection) signals drift, previously generated normal samples
are replayed together with the new data to update the model. This dynamic
approach leverages the VAE branch to mitigate concept drift.

### Required arguments

- `--dataset`: name of the dataset to use.
- `--data_path`: path to the dataset directory.
- `--win_size`: sliding window size (default `100`).
- `--input_c`: number of input channels.
- `--output_c`: number of output channels.
- `--batch_size`: training batch size (default `256`).
- `--num_epochs`: training epochs (default `10`).
- `--lr`: learning rate for the Adam optimizer (default `1e-4`).
- `--k`: weighting factor for the association discrepancy losses (default `3`).
- `--anomaly_ratio`: anomaly ratio in training set (default `1.0`).
- `--model_save_path`: directory for checkpoints and results (default
  `checkpoints`).
- `--model_type`: `transformer` or `transformer_vae` (default
  `transformer_vae`).
- `--cpd_penalty`: penalty used by `ruptures` for change point detection
  (default `20`). A larger value results in fewer detected drifts.
- `--replay_horizon`: keep latent vectors for at most this many training
  steps when using the VAE model (default `None`).
- `--store_mu`: store `(mu, logvar)` pairs instead of sampled `z` for replay.
- `--freeze_after`: freeze the encoder after this many updates (default `None`).
- `--ema_decay`: apply EMA to encoder weights with this decay (default `None`).
- `--decoder_type`: choose decoder architecture: `mlp`, `rnn`, or `attention`.
- `--min_cpd_gap`: minimum separation between detected change points (default
  `30`).
- `--cpd_log_interval`: evaluate and print metrics only after this many CPD
  updates (default `20`).

After training, the script prints the number of updates triggered by CPD events.
Install the `ruptures` package (e.g., via `pip install ruptures`) so that these
change-point detection updates can occur. The `--cpd_penalty` argument controls
the sensitivity of this detection.

### Example

```bash
python incremental_experiment.py \
    --dataset SMD --data_path dataset/SMD \
    --input_c 38 --output_c 38
```

Use `--cpd_penalty` to tune how aggressively change points are detected. Larger
values, such as `--cpd_penalty 40`, will trigger fewer updates.

Training and evaluation artifacts are saved under `--model_save_path`.
Two figures, `f1_score.png` and `roc_auc.png`, visualize F1 score and ROC AUC
across the number of CPD-triggered updates. Starting with this version the
metrics are evaluated **whenever CPD causes a model update**, so each point
corresponds to a detected drift event rather than an epoch boundary.
F1 Score와 ROC AUC가 CPD 업데이트가 발생할 때마다 기록되어
`f1_score.png`와 `roc_auc.png` 파일로 저장됩니다.

## Visualization Utilities

The new module `utils/analysis_tools.py` provides helper functions for
qualitatively inspecting continual learning behavior.

- `plot_z_bank_tsne(model, loader, n_samples=500, save_path="z_bank_tsne.png")`
  projects latent vectors from the model's `z_bank` and from the provided
  dataset loader using **t-SNE**.
- `plot_z_bank_pca(model, loader, n_samples=500, save_path="z_bank_pca.png")`
  performs the same comparison with **PCA**, and `plot_z_bank_umap` relies on
  **UMAP** if available. Each helper saves a scatter plot contrasting original
  and replayed vectors.
- `visualize_cpd_detection(series, penalty=None, min_size=30, save_path="cpd_detection.png")`
  draws change points detected by `ruptures`. When `penalty` is ``None`` a
  heuristic based on series length is used, and `min_size` enforces a minimum
  gap between change points so the plot remains readable.
  dataset loader using t-SNE, saving a scatter plot that compares their
  distributions.
- `visualize_cpd_detection(series, penalty=20, save_path="cpd_detection.png")`
  draws change points detected by `ruptures` on top of a sequence so that you
  can confirm whether CPD corresponds to actual distribution shifts.
- A quick demo script `scripts/visualize_cpd_demo.py` generates a toy series and
  saves `cpd_demo.png`, `tsne_demo.png`, and `pca_demo.png` so you can verify
  that these utilities work without preparing a real dataset.
  This demo requires `numpy`, `scikit-learn`, `matplotlib`, and `ruptures`.

Directories in the provided `save_path` are created automatically, so you can
use paths such as `outputs/z_bank_tsne.png` without pre-creating the folder.

When using the VAE-based model (`--model_type transformer_vae`), these
visualizations are generated automatically at the end of training and saved
alongside the metric plots.


## Citation
If you find this repo useful, please cite our paper.

```
@inproceedings{
xu2022anomaly,
title={Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy},
author={Jiehui Xu and Haixu Wu and Jianmin Wang and Mingsheng Long},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=LzQQ89U1qm_}
}
```

## Contact
If you have any question, please contact wuhx23@mails.tsinghua.edu.cn.
