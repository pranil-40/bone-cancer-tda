# step10 setup check: confirms torch/torchvision are installed and reports whether
# CUDA is available (it isn't here - step10_cnn_models.py runs CPU-only).

import torch, torchvision
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")