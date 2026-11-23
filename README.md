# People Counting: OpenCV vs YOLOv8x

A comparative study of classical computer vision and deep learning approaches for frame-by-frame people counting in video sequences.

## Overview

This project evaluates two approaches to people counting:

- **OpenCV (Canny + Contours)** — a classical image-processing pipeline based on edge detection and contour analysis.
- **YOLOv8x** — a pretrained deep learning object detector using COCO weights to detect people.

Both methods are evaluated on the **Mall Dataset**, using the same test sequence of 1,200 frames.

The objective is to compare counting accuracy and computational performance under the selected experimental configuration.

## Dataset

The experiments use the Mall Dataset, which contains video frames captured in a shopping mall and ground-truth annotations of the number of people.

The dataset is downloaded automatically using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download("chaozhuang/mall-dataset")
```

The original dataset is available on Kaggle:

[ Mall Dataset — Kaggle ](https://www.kaggle.com/datasets/chaozhuang/mall-dataset)

The experiment uses frames **801–2000** (1,200 frames in total). The first 800 frames are excluded from the test sequence.

The ground-truth annotations are loaded from `mall_gt.mat`.

## Methods

### 1. OpenCV: Canny + Contour Analysis

The classical image-processing pipeline consists of:

1. Conversion to grayscale.
2. Gaussian blur.
3. Canny edge detection (thresholds: 30 and 100).
4. Morphological dilation using a 3×3 kernel.
5. External contour detection.
6. Filtering contours by area (greater than 30 pixels).

The final count is determined by the number of contours satisfying the area threshold.

### 2. YOLOv8x

The second approach uses the pretrained YOLOv8x model:

- Model: `yolov8x.pt`
- Pretrained weights: COCO
- Input image size: 640
- Confidence threshold: 0.25
- Target class: `person`
- Device: GPU (CUDA), if available; otherwise CPU.

No additional training or fine-tuning on the Mall Dataset is performed.

## Evaluation Metrics

The methods are evaluated using the following metrics:

- **MAE (Mean Absolute Error)** — average absolute difference between predicted and ground-truth counts.
- **RMSE (Root Mean Squared Error)** — penalizes larger counting errors more strongly.
- **Bias** — measures systematic overcounting or undercounting.
- **FPS (Frames Per Second)** — measures processing speed.

The FPS measurements exclude image loading time. Test images are preloaded into RAM before benchmarking.

## Experimental Environment

The experiment is implemented in Python and designed to run in Google Colab.

Main libraries:

- OpenCV
- Ultralytics
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib
- KaggleHub

## Repository Contents

- `people_counting.ipynb` — Google Colab notebook containing the experimental pipeline.
- `metrics_comparison.png` — visualization of ground-truth counts, predictions, and absolute errors.
- `mall_counting_summary.csv` — summary of evaluation metrics.
- `mall_counting_detailed_results.csv` — frame-by-frame predictions and errors.

## How to Run

1. Open `people_counting.ipynb` in Google Colab.
2. Run the notebook cells sequentially.
3. The Mall Dataset will be downloaded automatically through KaggleHub.
4. The notebook will execute both counting methods.
5. Evaluation metrics will be calculated.
6. CSV files and the comparison plot will be generated.

A GPU runtime is recommended for YOLOv8x inference.

## Limitations

- The experiment uses one dataset and one test sequence.
- YOLOv8x is applied without fine-tuning on the target dataset.
- The OpenCV and YOLOv8x pipelines are benchmarked on different hardware types when CUDA is available (CPU and GPU, respectively).
- The experiment evaluates frame-level counts, not object tracking or unique visitor counting.

Therefore, the reported FPS values characterize the tested configurations rather than an isolated comparison of algorithmic complexity.

## Purpose

This repository accompanies a research paper on the application of computer vision to people counting in video recordings.

It provides the implementation and experimental outputs needed to inspect the methodology and reproduce the main evaluation procedure.
