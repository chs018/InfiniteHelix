import torch

print("=" * 60)
print("GPU VERIFICATION")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"GPU Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("GPU Name: CPU Only")
    print("GPU Memory: N/A")

print("=" * 60)
