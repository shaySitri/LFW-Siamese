# LFW-Siamese Neural Network for Face Similarity 🔍👤

[![Jupyter Notebook](https://img.shields.io/badge/Notebook-LFW_Siamese.ipynb-orange?logo=jupyter)](https://github.com/shaySitri/LFW-Siamese/blob/main/LFW_Siamese.ipynb)

This repository implements a **Siamese Neural Network** for face similarity detection using the [Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/) dataset. The network is trained to classify whether two face images belong to the same person, using one-shot learning with Siamese architecture.

---

## 📄 Project Overview
The **Siamese Neural Network** utilizes paired inputs processed through identical subnetworks to generate embeddings that capture facial characteristics. These embeddings are then compared using a distance metric to determine similarity, making the model effective for face verification tasks.

---

## 📊 Dataset and Preprocessing
- **Dataset**: [LFW - Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/) with 13,233 images of 5,749 people.
- **Data Characteristics**:
  - Public figures in grayscale images.
  - Each individual has an average of 2 images.
  - Some individuals have many more images (e.g., George W. Bush with 530 images).
- **Preparation**:
  - **Training Set**: 2,200 image pairs.
  - **Test Set**: 1,000 image pairs.
  - **Split**: Validation split ensures individuals in training don’t appear in validation.

---

## 🧩 Model Architecture and Hyperparameters
- **Input**: 105x105 grayscale images, aligned with the input size in [reference paper](https://openreview.net/pdf?id=S1v4N2eFO6).
- **Network**:
  - **Convolutional Layers** for feature extraction.
  - **Embedding Layer** to map inputs to a feature space.
- **Distance Metric**: Euclidean distance to calculate similarity between embeddings.
- **Loss Function**: Binary Cross-Entropy (BCE) for binary classification (same/different person).
- **Optimization**:
  - **Optimizer**: SGD and AdamW (tested for comparison).
  - **Learning Rate Schedule**: Exponentially decayed learning rate with initial momentum.

### Hyperparameters
- **Batch Size**: 128
- **Learning Rate**: Varied with experiments, adjusted by epoch
- **Early Stopping**: Training stopped if no improvement after 20 epochs.

---

## 🧪 Experimental Results
The report highlights multiple experiments with variations in learning rate, batch size, and input size. Key findings include:
- **Best Configuration**: Achieved high AUC and minimized false positives with smaller batch sizes and a 105x105 input image size.
- **Augmentation Techniques**: Tested augmentations like horizontal flip, rotation, and color jitter to enhance model generalization, with significant improvements in validation accuracy.

---

## 🛠️ Tools and Technologies Used
- **Python** 🐍
- **Deep Learning Framework**: Pytorch for model implementation
- **Data Handling**: NumPy
- **Evaluation**: AUC-ROC, accuracy, precision, and recall metrics

---

## 🔍 Key Findings
- **Optimal Thresholding**: Adjusting similarity thresholds minimized false positives.
- **Augmentation Benefits**: Data augmentation increased model stability and accuracy.
- **Effective for One-Shot Learning**: The Siamese network successfully generalized to unseen individuals using minimal examples.
