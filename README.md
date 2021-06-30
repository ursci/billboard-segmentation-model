# urban-sciences-model

# Requirements
- python3.8
- CUDA10.1
- torch 1.7.1
- torchvision 0.8.2
- segmentation_models_pytorch

# How it works

```sh
pip install requirements.txt
```

```sh
git submodule update --init --recursive
```

# Start notebook

```sh
jupyter lab # or jupyter notebook
```

# Prediction

You can get inferenced PIL image below!

```python
from inference import get_masked_pil_img
output = get_masked_pil_img("path/to/file.jpg")
```


