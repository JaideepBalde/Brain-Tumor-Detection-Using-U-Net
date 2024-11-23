Brain Tumor Segmentation using U-Net
This project implements brain tumor segmentation using the U-Net deep learning architecture. It uses the BRATS 2017/2019 datasets, which include multimodal MRI scans and corresponding ground truth segmentation masks. The goal is to accurately segment tumor regions from brain MRI scans.

Datasets
The following datasets are used for training and evaluation:

BRATS 2017: A dataset of multimodal MRI scans (FLAIR, T1, T2, T1c) of brain tumors with corresponding ground truth segmentation masks.
BRATS 2019: An extended version of the BRATS 2017 dataset, providing more diverse data for model training and testing.
Project Features
Data Preprocessing: The MRI scans are preprocessed, including resizing, normalization, and augmentation to enhance model performance.
U-Net Model: The segmentation task is performed using the U-Net model, which consists of an encoder-decoder architecture with skip connections to preserve spatial features and enable precise localization of the tumor.
Training & Evaluation: The model is trained with augmented data, and evaluated using metrics such as the Dice Coefficient for segmentation accuracy.
Visualization: Visual comparisons of predicted segmentation masks with ground truth masks are provided to assess model performance.
Technologies Used
Deep Learning Framework: TensorFlow/Keras
Libraries: NumPy, Matplotlib, OpenCV
Dataset: BRATS 2017/2019
Setup & Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation
Install required dependencies:

Copy code
pip install -r requirements.txt
Download the BRATS 2017/2019 datasets from BRATS Challenge and place them in the data folder.

Usage
Preprocess the data:

The preprocessing scripts will resize, normalize, and augment the dataset to prepare it for training.
Train the U-Net model:

Use the provided training script to train the U-Net model on the preprocessed dataset.
Copy code
python train.py
Evaluate the model:

After training, evaluate the model performance on a validation set using metrics like the Dice Coefficient.
Visualize the results:

View the predicted segmentation masks alongside the ground truth for model performance evaluation.
Results
The model can successfully segment tumor regions from brain MRI scans with a high degree of accuracy, as demonstrated by the Dice Coefficient metric.
