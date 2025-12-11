# CNN Depth Comparison on CIFAR-10

This repository contains the complete implementation, documentation, and presentation materials for the project:

"A Practical Tutorial on Convolutional Neural Networks:  
How Network Depth Affects Image Classification Performance on CIFAR-10"

This project was developed as part of a university coursework submission. It includes a full academic report, a recorded video tutorial with transcript, and all source code required to reproduce the experiments.

---

## Project Objective

The main objective of this project is to demonstrate how the depth of a Convolutional Neural Network (CNN) affects image classification performance. Two different CNN architectures were designed and compared:

- A Shallow CNN with a single convolutional block.
- A Deep CNN with multiple convolutional blocks and dropout regularisation.

Both models were trained under identical conditions using the CIFAR-10 dataset in order to ensure a fair and meaningful comparison.

---

## Dataset

The CIFAR-10 dataset is a benchmark image classification dataset consisting of:

- 60,000 colour images
- Image size: 32 × 32 pixels
- 10 object classes:
  Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- 50,000 training images
- 10,000 test images

The dataset is automatically loaded using the built-in Keras function:

keras.datasets.cifar10

No manual dataset download is required.

---

## Model Summary

### Shallow CNN
- One convolutional layer
- One max-pooling layer
- One dense hidden layer
- Softmax output layer
- Test accuracy: 63.57%

### Deep CNN
- Multiple convolutional blocks
- Increasing number of filters (32 → 64 → 128)
- Max-pooling after each block
- Dropout layer for regularisation
- Dense hidden layer
- Softmax output layer
- Test accuracy: 75.42%

The results confirm that increasing network depth significantly improves classification performance.

---

## Repository Structure

- cnn_depth_filter_cifar10.ipynb  
  Jupyter Notebook containing the full implementation of both CNN models, including data loading, training, evaluation, and visualisation.

- Tutorial Report.pdf  
  The complete written tutorial submitted for assessment.

- video.mp4  
  The recorded video explanation of the project.

- video_transcript.pdf  
  The full transcript of the recorded video tutorial.

- shallow_cnn_model.keras  
  Saved trained shallow CNN model.

- deep_cnn_model.keras  
  Saved trained deep CNN model.

- README.md  
  This documentation file.

---

## Software Requirements

To run this project locally, the following software is required:

- Python 3.10 or higher
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Installation Instructions

Open Command Prompt and run:

pip install tensorflow numpy matplotlib scikit-learn notebook

---

## How to Run the Project

1. Open Command Prompt.
2. Navigate to the project folder:
   cd Desktop\CNN_Tutorial_Project
3. Launch Jupyter Notebook:
   jupyter notebook
   or
   python -m notebook
4. Open the file:
   cnn_depth_filter_cifar10.ipynb
5. Click:
   Kernel → Restart & Run All

This will automatically train both CNN models and reproduce all results and figures shown in the report.

---

## Accessibility Considerations

- High-contrast plots were used for better readability.
- The report is structured using clear headings for screen reader compatibility.
- The video tutorial is supported by a full transcript.

---

## Author Information

Name: Sanaan Ashfaq 
University: Hertfordshire
Course: MSc DataScience 
Year: 2025  

---
## Github

GitHub Repository Link:
[https://github.com/SanaanAshfaq/cnn_depth_filter_cifar10] 

## Video

Video Link:
[https://drive.google.com/file/d/1f_-RBEhX6NhgMS_UGIibgss9b9fbdi-G/view?usp=sharing] 

## License

This project is released under the MIT License. You are free to use, modify, and distribute this project for educational and research purposes with proper attribution.
