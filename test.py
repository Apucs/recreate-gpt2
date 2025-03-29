import torch

# Define the shape of sd_hf[k] and sd[k]
sd_hf_shape = (64, 128, 3)  # Conv1D layer (out_channels, in_channels, kernel_size)
sd_shape = (128, 64)  # Linear layer (in_features, out_features)

# Create tensors with these shapes
sd_hf_k = torch.randn(sd_hf_shape)  # Example tensor for sd_hf[k]
sd_k = torch.randn(sd_shape)  # Example tensor for sd[k]

# Verify the condition sd_hf[k].shape[::-1] == sd[k].shape
print(f"sd_hf[k].shape[::-1]: {sd_hf_k.shape[::-1]}")
print(f"sd[k].shape: {sd_k.shape}")
assert sd_hf_k.shape[::-1] == sd_k.shape  # This should pass as the shapes are compatible
