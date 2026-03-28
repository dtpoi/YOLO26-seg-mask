# MaskYOLO - Panorama Mask Batch Generator

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-AGPL--3.0-green)
![Python](https://img.shields.io/badge/python-3.10+-orange)

**Panorama object mask generator based on YOLO26-seg instance segmentation**

[中文](README_.md) / English
</div>

---

## Introduction

MaskYOLO is a specialized tool for generating object masks from 360° panoramic images.

Built on Ultralytics YOLO26 instance segmentation, it automatically detects and extracts specified targets (people, vehicles, etc.) from panoramic images, outputting high-quality binary mask images that integrate seamlessly with downstream rendering pipelines such as 3DGS (3D Gaussian Splatting).

### Features

- **High-precision instance segmentation** — Powered by YOLO26-seg, multi-class detection supported
- **Panorama-optimized pipeline** — ERP → Cubemap → YOLO inference → ERP, avoids recognition degradation caused by panoramic distortion
- **Modern GUI** — Built with Flet, supports real-time Chinese/English language switching
- **Custom filename suffix** — Append user-defined suffixes to output filenames

---

## Screenshot

![UI](examples/UI.png)

---

## Dependencies

| Dependency | Version | Description |
|------------|---------|-------------|
| [ultralytics](https://github.com/ultralytics/ultralytics) | ≥ 8.4.0 | YOLO26 model inference |
| [opencv-python](https://github.com/opencv/opencv) | ≥ 4.8.0 | Image processing and mask post-processing |
| [numpy](https://github.com/numpy/numpy) | ≥ 1.24.0 | Numerical computation |
| [py360convert](https://github.com/sunset1995/py360convert) | ≥ 0.1.0 | ERP ⇔ Cubemap projection conversion |
| [flet](https://github.com/flet-dev/flet) | ≥ 0.25.0 | GUI framework |
| [flet-desktop](https://github.com/flet-dev/flet) | ≥ 0.25.0 | Flet desktop support |
| [torch](https://github.com/pytorch/pytorch) | ≥ 2.0.0 | Deep learning framework (CUDA 12.4 optimized) |
| [torchvision](https://github.com/pytorch/pytorch) | ≥ 0.15.0 | Computer vision utilities |

> **Tip**: On first run, the program will automatically download `yolo26n-seg.pt` (~7 MB) from the Ultralytics official source. If download fails, manually obtain it from [Ultralytics Assets](https://github.com/ultralytics/assets/releases) and place it in the project root.

---

## 🌟 Environment Setup (UV)

```bash
# Clone the repo
git clone https://github.com/dtpoi/YOLO26-seg-mask.git
cd YOLO26-seg-mask

# Sync dependencies (automatically creates .venv and installs all packages)
uv sync

# Run the program
uv run python main.py
```

> **Note**: On Windows, `uv sync` automatically configures PyTorch with CUDA 12.4 for optimal GPU inference. To force CPU-only mode, set the environment variable `CUDA_VISIBLE_DEVICES=` before running.

---

## 🌟 Usage

### 1. Launch the Program

```bash
uv run python main.py
```

### 2. Configure Paths

- **Input Path** — Folder containing panoramic images (ERP format, `.jpg` / `.png`) or flat images
- **Output Path** — Destination folder for generated mask images

### 3. Select Target Classes

Check the desired object classes in the right panel (multi-select supported):

| Class ID | Name | Description |
|:--------:|------|-------------|
| 0 | person | Human figures |
| 1 | bicycle | |
| 2 | car | |
| 3 | motorcycle | |
| 5 | bus | |
| 7 | truck | |
| 8 | boat | |

### 4. Adjust Parameters (Optional)

| Parameter | Default | Description |
|-----------|:------:|------------|
| Dilation | 9 | Mask edge expansion in pixels, larger = wider coverage (recommended 7–11) |
| Inference Size | 1280 | YOLO inference resolution, higher = more detailed but slower (recommended 1280 / 1920) |
| Confidence | 0.15 | Detection threshold, lower = more sensitive but may cause false positives (recommended 0.10–0.25) |
| Cubemap Width | 1024 | Width per cubemap face in pixels, panorama mode only (recommended 1024 / 1536) |
| Batch Inference | ✅ On | Packs multiple faces for GPU inference to maximize utilization; turn off if VRAM is limited |
| Bypass Panorama Split | ❌ Off | Skips Cubemap conversion; suitable for flat/regular photos |
| Invert Mask | ✅ On | When on → target is black / background is white (for 3DGS mask layer) |
| Add Suffix | ❌ Off | Appends a custom suffix to output filenames (default `_mask`) |

### 5. Start Processing

Click **Start** to begin. The progress bar updates in real time. Click **Stop** at any moment to interrupt.

---

## Demo

### Person Mask (person)

| Original | Mask |
|:--------:|:----:|
| ![Original](examples/e1.jpg) | ![Person Mask](examples/mask1.png) |

### Vehicle Mask (car / truck)

| Original | Mask |
|:--------:|:----:|
| ![Original](examples/e1.jpg) | ![Vehicle Mask](examples/mask2.png) |

---

## Technical Details

### Panorama Processing Pipeline

```
ERP Panorama → ERP to Cubemap (6 faces) → Batch YOLO Inference → Merge Masks → Cubemap to ERP → Output
```

Panoramic images use Equirectangular Projection (ERP), which suffers from severe spherical distortion. Applying a standard CNN directly yields poor results. This project uses a classic Cubemap strategy:

1. **ERP → Cubemap** — Uses `py360convert` to unfold the spherical panorama into 6 planar cube faces
2. **YOLO Segmentation** — Runs YOLO26-seg instance segmentation independently on each face
3. **Batch Inference** — Packages all 6 faces into a single batch for the GPU to maximize utilization
4. **Cubemap → ERP** — Merges masks from all 6 faces and projects back to the original ERP format
5. **Post-processing** — Applies dilation to extend mask edges, avoiding residual artifacts from cutouts

### Flat Image Mode

When **Bypass Panorama Split** is enabled, images are fed directly to YOLO without projection conversion, skipping the CPU-only bottleneck of `py360convert`. This is ideal for regular flat photos.

---

## Project Structure

```
maskYOLO/
├── main.py               # Program entry point
├── pyproject.toml        # UV project configuration
├── uv.lock               # Dependency lock file
├── requirements.txt      # pip-compatible dependency list
├── assets/
│   └── fonts/
│       └── SarasaUiSC-Regular.ttf   # Sarasa Gothic UI font
├── examples/             # Sample images
│   ├── UI.png            # UI screenshot
│   ├── e1.jpg           # Original image
│   ├── mask1.png        # Person mask
│   └── mask2.png        # Vehicle mask
└── yolo26n-seg.pt        # YOLO26-seg pretrained weights (auto-downloaded)
```

---

## License

This project is open-source under **AGPL-3.0**. It depends on [ultralytics](https://github.com/ultralytics/ultralytics), which is also licensed under AGPL-3.0. Please comply with the respective license terms. For commercial closed-source usage, please contact Ultralytics for a commercial license.

---

## Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLO26 model
- [py360convert](https://github.com/sunset1995/py360convert) — Panorama projection conversion
- [Flet](https://flet.dev/) — Python GUI framework
- [Sarasa Gothic](https://github.com/be5invis/Sarasa-Gothic) — Sarasa Gothic font

---

## 📋 TodoList

- [ ] Compile an EXE or folder with a one click
- [ ] Replace py360convert with a panoramic conversion library that supports GPU acceleration to further improve the processing speed of panoramic images

---
<div align="center">

**If you find this useful, give it a ⭐ Star!**

</div>
