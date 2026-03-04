🫁 Lung Cancer Detection using Convolutional Neural Networks

An end-to-end deep learning system for detecting lung cancer from CT scan images using a custom Convolutional Neural Network (CNN). The project uses the IQ-OTH/NCCD lung CT dataset and demonstrates the full machine learning workflow from data preprocessing and model training to deployment using Streamlit.

📌 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates. This project aims to develop a deep learning model capable of classifying CT scan images as Normal or Cancerous.

The system processes CT scan images, extracts visual features using a CNN, and predicts the probability of lung cancer. The trained model is integrated into a Streamlit web application that allows users to upload CT images and receive real-time predictions.

🧠 Key Features

Custom Convolutional Neural Network (CNN) architecture

CT image preprocessing pipeline

Binary classification: Normal vs Cancer

Performance evaluation using multiple metrics

Model deployment with Streamlit

Interactive interface for image upload and prediction

End-to-end machine learning pipeline

📊 Dataset

Dataset used: IQ-OTH/NCCD Lung Cancer Dataset

The dataset contains 1190 annotated CT scan slices divided into three categories:

Class	Number of Images
Normal	416
Benign	120
Malignant	561

For this project, the dataset was converted into a binary classification task:

Normal

Cancer (Benign + Malignant)

Images were resized to 224 × 224 pixels and normalized before training.

🏗 Model Architecture

The model is a custom CNN architecture consisting of:

Convolutional layers for feature extraction

Batch normalization for training stability

Max pooling layers for spatial reduction

Dropout layers to prevent overfitting

Fully connected layers for classification

Sigmoid activation for binary prediction

⚙️ Technologies Used

Python

TensorFlow / Keras

NumPy

Scikit-learn

Matplotlib / Seaborn

Streamlit

📈 Model Performance

After training the CNN model:

Metric	Value
Training Accuracy	~99.7%
Validation Accuracy	~98.1%
Loss	Low and stable

🚀 Streamlit Web Application

A Streamlit interface was built to simulate real-world deployment.

Features

Upload CT scan image

Automatic preprocessing

Real-time prediction

Display prediction confidence

📂 Project Structure
lung-cancer-cnn/
│
├── dataset/
│   ├── cancer/
│   └── normal/
│
├── notebooks/
│   └── model_training.ipynb
│
├── model/
│   └── lung_cancer_cnn_model.keras
│
├── app.py
│
├── requirements.txt
│
└── README.md
▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/yourusername/lung-cancer-cnn.git
cd lung-cancer-cnn
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py

Then open the provided local URL in your browser.

⚠️ Disclaimer

This project is developed for educational and research purposes only.
It is not intended for clinical or medical diagnosis.

📌 Future Improvements

Multi-class classification (Normal / Benign / Malignant)

Grad-CAM visualization for model interpretability

Larger CT scan datasets

Model deployment on cloud platforms

👨‍💻 Author

Jatin Dahiya
B.Tech Biotechnology

Interested in AI in Healthcare, Bioinformatics, and Medical Imaging.
