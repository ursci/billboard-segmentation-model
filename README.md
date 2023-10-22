# urban-sciences-model

## Requirements
- python3.8
- CUDA10.1
- torch 1.7.1
- torchvision 0.8.2
- segmentation_models_pytorch

## How it works

```sh
pip install -r requirements.txt
```

```sh
git submodule update --init --recursive
```

## Preparation

You should set `best_model_Unet_resnet50_epoch40.pth` into notebooks directory, if you want to use a pre-trained model. Also you can get the weight from [here](https://huggingface.co/pomcho555/billboard-segmentation-model/blob/main/best_model_Unet_resnet50.pth).

## Start notebook

```sh
jupyter lab # or jupyter notebook
```

## Prediction

You can get the inferenced PIL image below!

```python
from inference import get_masked_pil_img
output = get_masked_pil_img("path/to/file.jpg")
```

## Original paper

[Quantifying urban streetscapes with deep learning: focus on aesthetic evaluation
](https://arxiv.org/abs/2106.15361)

[Dataset](https://paperswithcode.com/paper/quantifying-urban-streetscapes-with-deep)

## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{kumakoshi2021quantifying,
      title={Quantifying urban streetscapes with deep learning: focus on aesthetic evaluation}, 
      author={Yusuke Kumakoshi and Shigeaki Onoda and Tetsuya Takahashi and Yuji Yoshimura},
      year={2021},
      eprint={2106.15361},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


