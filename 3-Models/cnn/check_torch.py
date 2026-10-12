# Checks that torch/torchvision are installed and whether CUDA is available
# (it isn't on my machine, so cnn.py runs on CPU).

import torch, torchvision
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")