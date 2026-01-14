Customer Churn Prediction Using Deep Learning (ANN)

A deep learning–based customer churn prediction system built using Artificial Neural Networks (ANN) to identify customers likely to churn in a banking domain.
The project focuses on model design, optimization, and evaluation using Keras and TensorFlow.

🚀 Project Overview

Customer churn is a critical challenge in the banking industry. This project applies a feed-forward neural network to predict churn based on customer behavior and attributes.
Advanced activation functions and regularization techniques are used to improve learning stability and generalization.

🧠 Key Highlights

Deep learning–based churn prediction using ANN

Multiple hidden layers with LeakyReLU & PReLU activations

Dropout regularization to reduce overfitting

Adam optimizer for faster and stable convergence

Achieved ~90% accuracy on the test dataset

🛠️ Tech Stack

Language: Python

Deep Learning: TensorFlow, Keras

Data Processing: Pandas, NumPy

Model Type: Feed-forward Artificial Neural Network (ANN)

📁 Project Structure
Customer-Churn-ANN/
│
├── ANNGPU.ipynb              # Jupyter notebook for model development & experiments
├── churn_modelling_ann.py    # Python script for ANN model training and evaluation
├── Churn_Modelling.csv       # Dataset used for churn prediction
├── requirements.txt          # Project dependencies
├── .gitignore
└── .idea/                    # IDE configuration files

⚙️ Model Architecture

Input layer based on processed customer features

Multiple hidden layers with:

LeakyReLU & PReLU activations (to handle vanishing gradients)

Dropout for regularization

Output layer for churn classification

Optimized using Adam optimizer

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Training Script
python churn_modelling_ann.py

3️⃣ (Optional) Explore the Notebook
jupyter notebook ANNGPU.ipynb

📊 Dataset

Source: Banking customer churn dataset

Target Variable: Customer churn (Yes / No)

Features: Customer demographics, account details, and usage patterns

(Dataset included for educational and modeling purposes)

📈 Results

Achieved ~90% accuracy on the test dataset

Improved learning stability using advanced activation functions

Reduced overfitting with Dropout regularization

📌 Use Cases

Customer retention analytics

Churn risk prediction

Business decision support in banking & finance

🔮 Future Enhancements

Hyperparameter tuning (Grid / Random Search)

Model explainability (SHAP / feature importance)

Deployment as a REST API

Comparison with traditional ML models

👤 Author

Built as a hands-on deep learning project to demonstrate ANN modeling, optimization techniques, and real-world churn prediction.
