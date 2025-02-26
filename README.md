# IDOL: On the Identification of Temporally Causal Representation with Instantaneous Dependence (ICLR-2025)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 2.3.1](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

## Motivation
Temporally causal representation learning aims to identify the latent causal process from time series observations, but most methods require the assumption that the latent causal processes do not have instantaneous relations. Although some recent methods achieve identifiability in the instantaneous causality case, they require either interventions on the latent variables or grouping of the observations, which are in general difficult to obtain in real-world scenarios. To fill this gap, we propose an **ID**entification framework for instantane**O**us **L**atent dynamics (**IDOL**) by imposing a sparse influence constraint that the latent causal processes have sparse time-delayed and instantaneous relations. Three different data generation processes with time-delayed and instantaneous dependencies as shown in Figure 1.
<p align="center">
<img src=".\pic\motivation.png" height = "300" alt="" align=center />
<br><br>
<b>Figure 1.</b> (a) iCITRIS requires intervention variables $i_t$ for latent variables (the gray nodes with blue rim.). (b) G-CaRL assumes that observed variables can be grouped according to which latent variables they are connected to (the blue dotted lines), (c) IDOL requires a sparse latent causal process (the blue solid lines).
</p>

## Model
Based on theoretical results, we develop the **IDOL** model as shown in Figure 2, which is built on the variational inference to model the distribution of observations. To estimate the prior distribution and enforce the independent noise assumption, we devise the prior networks. Moreover, we employ a gradient-based sparsity penalty to promote the sparse causal influence. 
<p align="center">
<img src=".\pic\model.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 2.</b> The framework of the IDOL model. The encoder and decoder are used for the extraction of latent variables and observation reconstruction. The prior network is used for prior distribution estimation, and  $L_s$ denotes the gradient-based sparsity penalty. The solid and dashed arrows denote the forward and backward propagation.

## Requirements

- Python 3.8
- torch == 2.4.1
- tslearn==0.6.3
- reformer-pytorch==1.4.4
- einops == 0.4.0
- tqdm == 4.64.1

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## Data

We have already put the datasets in the .\dataset\ file.
## Reproducibility

To easily reproduce the results you can run the following commands:
```
cd realworld
bash ./scripts/forecasting/Human/Walking.sh
```
Multiple prediction lengths can be run at one time. 

And we provide explanations for the important parameters:
| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| root_path      | The root path of the data file (defaults to `./dataset/human/`)    |
| data_path      | The data file name (defaults to `Walking_all.npy`)                  |
| features       | The forecasting task (defaults to `M`). This can be set to `M`,`S`,`MS` (M : multivariate predict multivariate, S : univariate predict univariate, MS : multivariate predict univariate) |
| seq_len | Input sequence length of LSTD encoder (defaults to 125) |
| label_len | Start token length of LSTD decoder (defaults to 125) |
| pred_len | Prediction sequence length (defaults to 125) |
| des | exp description |
| itr | experiments times |
| test_bsz | Batch size in test |
| train_epochs | Epochs in train |

More parameter information please refer to `main.py`.


## <span id="resultslink">Results</span>

The main results are shown in Table 1.
<p align="center">
<b>Table 1.</b> MSE and MAE results on the different motions.
<img src=".\pic\results.png" height = "500" alt="" align=center />
<br><br>
</p>

## <span id="resultslink">Visualization</span>

To visualize the Human dataset, you can run the following command:
```
python visual.py -dataset Walking_all -pred_len 125
```
Partial visualization results are shown in Figure 3.
<p align="center">
<img src=".\pic\visual.gif" alt="" align=center />
<br><br><b>Figure 3.</b> The illustration of visualization of Walking. The red lines denote the ground-truth motions, the green lines denote the prediction motions of IDOL. 
</p>

## <span id="citelink">Citation</span>
If you find this repository useful in your research, please consider citing the following papers:

```
@inproceedings{
li2025on,
title={On the Identification of Temporal Causal Representation with Instantaneous Dependence},
author={Zijian Li and Yifan Shen and Kaitao Zheng and Ruichu Cai and Xiangchen Song and Mingming Gong and Guangyi Chen and Kun Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2efNHgYRvM}
}
```
