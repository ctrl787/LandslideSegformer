# LandslideSegFormer

**Lightweight Landslide Segmentation with Enhanced Mix Transformer**

A lightweight semantic segmentation model designed for landslide detection from remote sensing imagery.

## Architecture

LandslideSegFormer combines the following components:

- **EMiT Encoder**: Enhanced Mix Transformer encoder (B0-level, ~3.7M parameters)
  - Efficient self-attention with spatial reduction
  - Mix-FFN with depthwise convolutions
  - Local enhancement modules

- **Parallel CNN Branch**: Complementary local feature extraction for edge preservation

- **Non-Local Block**: Global context modeling for capturing long-range dependencies

- **GHAD Decoder**: Gated Hierarchical Attention Decoder
  - Lightweight MLP-based design
  - Multi-scale feature fusion with learnable gates
  - Channel attention at each scale

## Model Highlights

| Metric | Value |
|--------|-------|
| Parameters | ~16M |
| Input Size | 512×512 |
| Output | Binary segmentation mask |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python demo.py
```

## Usage

```python
import torch
from model import LandslideSegFormer

# Create model
model = LandslideSegFormer(num_classes=2)
model.eval()

# Inference
x = torch.randn(1, 3, 512, 512)
output = model(x)  # Shape: [1, 2, 512, 512]
```

## Architecture Diagram

```
Input Image (3×512×512)
        │
        ├──────────────────────┐
        │                      │
        ▼                      ▼
  ┌──────────┐          ┌──────────┐
  │   EMiT   │          │   CNN    │
  │ Encoder  │          │  Branch  │
  └────┬─────┘          └────┬─────┘
       │                     │
       │    ┌────────────────┘
       │    │
       ▼    ▼
  ┌──────────────┐
  │   Feature    │
  │   Fusion     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Non-Local   │
  │    Block     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │    GHAD      │
  │   Decoder    │
  └──────┬───────┘
         │
         ▼
   Output (2×512×512)
```

