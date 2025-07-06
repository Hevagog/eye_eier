# Eye Detection with Burn

This project implements an eye region detection model using the [Burn](https://github.com/tracel-ai/burn) deep learning framework in Rust. The model is designed to localize the eye region in human face images, using a dataset with YOLOv8-style bounding box annotations.

## Dataset

- **Source:** [Eye Detection Dataset on Kaggle](https://www.kaggle.com/datasets/icebearogo/eye-detection-dataset)

## Project Structure

- `src/`
    - `data.rs` — Dataset loading, preprocessing, and batching logic.
    - `model.rs` — Model architecture (convolutional neural network for bounding box regression).
    - `training.rs` — Training loop, configuration, and evaluation.
    - `constants.rs` — Image size and other constants.
    - `utils.rs` — Utility functions for image/label handling and visualization.

