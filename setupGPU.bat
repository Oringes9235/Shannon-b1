pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

