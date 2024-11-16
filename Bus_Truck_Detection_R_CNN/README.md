# R-CNN: Detecting Buses and Trucks in Images

[![Project Overview](https://img.shields.io/badge/Original-Paper-blue)](https://arxiv.org/abs/1311.2524) <!-- Link to the original paper -->

This project leverages the power of **Region-based Convolutional Neural Networks (R-CNN)** to detect **buses** and **trucks** in images. It showcases the simplicity and efficiency of R-CNN for object detection tasks, offering insights into its pipeline and implementation.

---

## 🚀 Features

- Detect buses and trucks with high accuracy.
- Utilizes pre-trained models for efficient feature extraction.
- Custom dataset preparation and annotation.
- Implements R-CNN for object detection.

---

## 📁 Project Structure

```plaintext
.
├── Dataset/               # Contains datasets and annotations
│   ├── df.csv             # Dataset preparation and augmentation
│   ├── images/            # images
│   │   ├──images
├── R_CNN.ipynb/           # Jupyter notebooks for experiments and visualization 
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies

## 🛠️ Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your_username/simple-rcnn-detection.git
   cd simple-rcnn-detection

