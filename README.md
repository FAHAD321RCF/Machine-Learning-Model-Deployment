House Price Prediction API
A machine learning web application that predicts house prices based on input features. This project demonstrates the complete pipeline of model training, deployment using Flask, Docker, and hosting on platforms like Heroku or AWS.

Features
Predict house prices based on 13 features using a trained Linear Regression model.
Exposes the model as a REST API.
Dockerized for containerization.
Ready for deployment on Heroku or AWS.

Tech Stack
Programming Language: Python
Libraries: Flask, Scikit-learn, Pandas, Numpy
Machine Learning Framework: Scikit-learn
Containerization: Docker
Deployment Options: Heroku, AWS

Project Structure:

.
├── app.py                 # Flask app serving the ML model
├── app_fastapi.py         # Optional FastAPI version of the app
├── train_model.py         # Script for training and saving the model
├── model.pkl              # Trained Linear Regression model
├── scaler.pkl             # Scaler used for data preprocessing
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration for the app
├── Procfile               # Heroku deployment configuration
└── README.md              # Project documentation

