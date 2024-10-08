# AdaptVis: Spatial Understanding in Vision-Language Models Requires Adaptive Attention

Code and datasets for "AdaptVis: Spatial Understanding in Vision-Language Models Requires Adaptive Attention".


This code is based on the code of, **What's "up" with vision-language models? Investigating their struggle with spatial reasoning** [[paper](https://arxiv.org/pdf/2310.19785)][[code](https://github.com/amitakamath/whatsup_vlms)].

<p align="center">
<img src="figures/main.png" width="800">
</p>


# Datasets
 The code to load and evaluate each dataset in `dataset_zoo/aro_datasets.py`. The Question and Answering data is in `prompt/`

# ScalingVis and AdaptVis

## Setting Up the environment

```
git clone https://github.com/shiqichen17/AdaptVis.git
mkdir data
mkdir output
pip install requirements.txt
```

## Downloading the data
The data all lives in `whatsup_vlms/data`, which is also where your models will go as they're downloaded.   

For all the datasets, setting `--download=True` (while running `python main_aro.py` or while instantiating the dataset directly, as mentioned later in this README) will download the data JSONs and images if the files don't already exist.

You can also download the data directly from [this Google Drive link](https://drive.google.com/drive/u/3/folders/164q6X9hrvP-QYpi3ioSnfMuyHpG5oRkZ).


## Running experiments scaling_vis and adapt_vis
You can fast implement an example by:
```
bash run.sh
```
### Argument
| Argument       | Example               | Description                                                                                   |
|----------------|-----------------------|-----------------------------------------------------------------------------------------------|
| `dataset`          | `Controlled_Images_A` | Specifies the dataset you want to evaluate. Can choose from `Controlled_Images_A, Controlled_Images_B..`. |
| `model`              | `llava1.5`            | Specifies the model you want to use.                                                          |
| `method`                | `scaling_vis`         | The method for evaluation. Can choose from `"scaling_vis"` or `"adapt_vis"`.                  |
| `weight`                   | `1.2`                 | Coefficient for Scaling_vis. Can set from `[0, 5, 0.8, 1.2, 1.5, 2.0]`.                       |
| `weight1`           | `0.5`                 | Coefficient for AdaptVis. Can set from `[0.5, 0.8]`.                                          |
| `weight2`          | `1.2`                 | Coefficient for AdaptVis. Can set from `[1.2, 1.5, 2.0]`.                                     |
| `th`                 | `0.3`                 | Threshold for AdaptVis.                                                                        |


# Citation
If you use this code or data, please consider citing our paper:
```
```
