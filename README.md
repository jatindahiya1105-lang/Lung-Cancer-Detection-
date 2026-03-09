🫁 Lung Cancer Detection using Convolutional Neural Networks

An end-to-end deep learning system for detecting lung cancer from CT scan images using a custom Convolutional Neural Network (CNN). The project uses the IQ-OTH/NCCD lung CT dataset and demonstrates the full machine learning workflow from data preprocessing and model training to deployment using Streamlit.

📌 Project Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Early detection significantly improves survival rates. This project aims to develop a deep learning model capable of classifying CT scan images as Normal or Cancerous.

The system processes CT scan images, extracts visual features using a CNN, and predicts the probability of lung cancer. The trained model is integrated into a Streamlit web application that allows users to upload CT images and receive real-time predictions.

🧠 Key Features

-Custom Convolutional Neural Network (CNN) architecture

-CT image preprocessing pipeline

-Binary classification: Normal vs Cancer

-Performance evaluation using multiple metrics

-Model deployment with Streamlit

-Interactive interface for image upload and prediction

-End-to-end machine learning pipeline

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

Images were resized to 244 × 244 pixels and normalized before training.

🏗 Model Architecture

The model is a custom CNN architecture consisting of:

Convolutional layers for feature extraction

Batch normalization for training stability

Max pooling layers for spatial reduction

Dropout layers to prevent overfitting

Fully connected layers for classification

Sigmoid activation for binary prediction

🔬 Methodology

The project was developed as an end-to-end deep learning pipeline for lung cancer detection using CT scan images. The workflow consists of data preprocessing, model development, training, evaluation, and deployment.

1. Dataset Preparation

The dataset used in this project is the **IQ-OTH/NCCD lung cancer CT scan dataset**, which contains 1190 annotated CT scan slices categorized as Normal, Benign, and Malignant.

For this project, the task was simplified into a binary classification problem:

- Normal
- Cancer (Benign + Malignant)

The dataset was divided into training, validation, and test sets to ensure proper evaluation of the model.

Images were resized to **244 × 244 pixels** and converted to RGB format to match the input requirements of the neural network.

Data loading and preprocessing were performed using `ImageDataGenerator` from TensorFlow/Keras, with preprocessing functions applied to normalize the pixel values.


2. Data Preprocessing

The following preprocessing steps were applied:

- Image resizing to 244 × 244
- Pixel normalization using MobileNetV2 preprocessing function
- Batch generation using ImageDataGenerator
- Separation into train, validation, and test sets
- Shuffle disabled during evaluation to maintain label order

This ensured consistent input formatting for the neural network.


3. Model Architecture

A Convolutional Neural Network (CNN) was used to extract spatial features from CT scan images.

The architecture consists of:

- Convolutional layers for feature extraction
- Activation functions (ReLU)
- Max pooling layers for dimensionality reduction
- Fully connected (Dense) layers for classification
- Softmax output layer for categorical prediction

The model was implemented using **TensorFlow and Keras Sequential API**.


4. Model Training

The model was trained using the following configuration:

- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Batch size: 4
- Input shape: (244, 244, 3)
- Class mode: categorical

Training was performed on the training dataset, while validation data was used to monitor performance and prevent overfitting.


5. Model Evaluation

The trained model was evaluated using the validation and test datasets.

Performance metrics included:

- Accuracy
- Loss
- Prediction confidence scores

The model achieved high validation accuracy, indicating good ability to distinguish between normal and cancerous CT scans.


6. Deployment

To demonstrate real-world usability, the trained model was deployed using **Streamlit**.

The web application allows users to:

- Upload CT scan images
- Automatically preprocess the image
- Run the trained CNN model
- Display prediction results with confidence score

The model file is dynamically loaded at runtime, ensuring compatibility with cloud deployment environments.


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

-Training Accuracy	~99.7%

-Validation Accuracy	~98.1%

-Loss	Low and stable

🚀 Streamlit Web Application

A Streamlit interface was built to simulate real-world deployment.

Features

-Upload CT scan image

-Automatic preprocessing

-Real-time prediction

-Display prediction confidence

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
-Run the Streamlit App
https://lung-cancer-detection-cnn-jd.streamlit.app/.py

Then upload the CT Scan Image and access you result.

⚠️ Disclaimer

This project is developed for educational and research purposes only.
It is not intended for clinical or medical diagnosis.

📌 Future Improvements

-Multi-class classification (Normal / Benign / Malignant)

-Grad-CAM visualization for model interpretability

-Larger CT scan datasets

-Model deployment on cloud platforms

👨‍💻 Author

Jatin Dahiya

B.Tech Biotechnology

Interested in AI in Healthcare, Bioinformatics, and Medical Imaging.
