"""
Demo script for LandslideSegFormer

Verifies the model architecture by running forward pass.
"""

import torch
from model import LandslideSegFormer


def main():
    print("=" * 60)
    print("LandslideSegFormer Demo")
    print("=" * 60)
    
    # Create model
    model = LandslideSegFormer(num_classes=2, in_channels=3)
    model.eval()
    
    # Test input
    x = torch.randn(1, 3, 512, 512)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*40}")
    print("Model Statistics")
    print(f"{'='*40}")
    print(f"Total parameters:     {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Component breakdown
    print(f"\n{'='*40}")
    print("Component Breakdown")
    print(f"{'='*40}")
    
    encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    cnn_params = sum(p.numel() for p in model.cnn_branch.parameters()) / 1e6
    fusion_params = sum(p.numel() for p in model.fusion.parameters()) / 1e6
    nonlocal_params = sum(p.numel() for p in model.non_local.parameters()) / 1e6
    decoder_params = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    
    print(f"EMiT Encoder:    {encoder_params:.2f}M")
    print(f"CNN Branch:      {cnn_params:.2f}M")
    print(f"Feature Fusion:  {fusion_params:.2f}M")
    print(f"Non-Local Blocks:{nonlocal_params:.2f}M")
    print(f"GHAD Decoder:    {decoder_params:.2f}M")
    
    # Test different input sizes
    print(f"\n{'='*40}")
    print("Multi-scale Input Test")
    print(f"{'='*40}")
    
    for size in [256, 384, 512]:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            out = model(x)
        print(f"Input: {size}x{size} -> Output: {out.shape[2]}x{out.shape[3]}")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
