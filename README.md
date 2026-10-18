# Employee Safety Detection Using YOLO11

A computer vision project for construction-site safety monitoring using a custom-trained YOLO11 model and worker tracking.

## Overview

This project analyzes construction-site video footage to detect personal protective equipment (PPE) and identify potential safety concerns.

## Features

- Detects helmets and non-helmet cases
- Detects gloves, shoes, safety vests, and bare arms
- Detects and tracks workers across video frames
- Processes video footage and generates annotated output
- Uses a custom-trained YOLO11 model

## Technologies

- Python
- Ultralytics YOLO11
- OpenCV
- PyTorch
- Google Colab

## Detection Classes

`Gloves`, `Helmet`, `Non-Helmet`, `Person`, `Shoes`, `Vest`, `bare-arms`

## Repository Structure

- `construction_safety_v9_sh17.py` — Main Python script
- `employee_safety.ipynb` — Colab notebook
- `best.pt` — Trained model weights
- `videos/` — Test footage and annotated output

## Getting Started

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Python script:

```bash
python construction_safety_v9_sh17.py
```

Ensure that the trained model and input video are available at the paths expected by the script. Update the paths if necessary.

Alternatively, open `employee_safety.ipynb` in Google Colab.

## Limitations

Detection performance can vary with lighting, camera angle, object size, and occlusion. PPE detections may flicker when workers overlap. This project is a prototype and should not replace professional safety inspections.

## Future Improvements

- Improve PPE detection during occlusion
- Reduce detection flickering
- Improve association of PPE with individual workers
- Evaluate performance across different environments

## Author

Nimra Faryal

Computer Vision | Python | AI/ML
