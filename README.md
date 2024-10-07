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
<table>
  <thead>
    <tr>
      <th style="width: 20%;">Argument</th>
      <th style="width: 20%;">Example</th>
      <th style="width: 60%;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>--dataset</code></td>
      <td><code>Controlled_Images_A</code></td>
      <td>Dataset for evaluation (<code>Controlled_Images_A</code>, <code>Controlled_Images_B</code>, etc.).</td>
    </tr>
    <tr>
      <td><code>--model</code></td>
      <td><code>llava1.5</code></td>
      <td>Model to use.</td>
    </tr>
    <tr>
      <td><code>--method</code></td>
      <td><code>scaling_vis</code></td>
      <td>Evaluation method (<code>scaling_vis</code> or <code>adapt_vis</code>).</td>
    </tr>
    <tr>
      <td><code>--weight</code></td>
      <td><code>1.2</code></td>
      <td>Scaling coefficient, options: <code>[0.5, 0.8, 1.2, 1.5, 2.0]</code>.</td>
    </tr>
    <tr>
      <td><code>--weight1</code></td>
      <td><code>0.5</code></td>
      <td>Coefficient for AdaptVis, options: <code>[0.5, 0.8]</code>.</td>
    </tr>
    <tr>
      <td><code>--weight2</code></td>
      <td><code>1.2</code></td>
      <td>Coefficient for AdaptVis, options: <code>[1.2, 1.5, 2.0]</code>.</td>
    </tr>
    <tr>
      <td><code>--th</code></td>
      <td><code>0.3</code></td>
      <td>Threshold for AdaptVis.</td>
    </tr>
  </tbody>
</table>




# Citation
If you use this code or data, please consider citing our paper:
```
```
