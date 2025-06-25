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

After training, the script prints the number of updates triggered by CPD events.
Install the `ruptures` package (e.g., via `pip install ruptures`) so that these
change-point detection updates can occur.

### Example

```bash
python incremental_experiment.py \
    --dataset SMD --data_path dataset/SMD \
    --input_c 38 --output_c 38
```

Training and evaluation artifacts are saved under `--model_save_path`.
A figure named `update_performance.png` will visualize validation loss and F1
score across the number of CPD-triggered updates.
Metrics are recorded at the end of each epoch, so each point corresponds to the
update count observed up to that epoch.
A figure named `update_performance.png` will visualize validation loss over the
number of CPD-triggered updates.

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
