# urban-sciences-model

# 環境
- python3.8
- CUDA10.1
- torch 1.7.1
- torchvision 0.8.2
- segmentation_models_pytorch

# 環境構築

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

# 第二次曲線セグメンテーション予測方法
下記でPILイメージが取得できます

```python
from inference import get_masked_pil_img
output = get_masked_pil_img("path/to/file.jpg")
```


