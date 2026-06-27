# 📩 Spam SMS Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Used-yellow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 🧠 Project Overview

This project is a **Machine Learning-based Spam SMS Detection system** that classifies SMS messages as:

- 🚫 **Spam**
- ✅ **Ham (Not Spam)**

It uses **Natural Language Processing (NLP)** techniques to clean and process text data, and a supervised machine learning model to make predictions.

The project also includes a **Flask web application** for real-time message classification.



## 🎯 Objective

The main goal of this project is to:
- Detect spam messages automatically
- Reduce unwanted messages
- Improve SMS filtering using AI/ML


## 🛠️ Technologies Used

- Python 🐍
- Pandas 📊
- NumPy 🔢
- Scikit-learn 🤖
- NLP (Natural Language Processing) 🧠
- Flask 🌐
- Jupyter Notebook 📓


## 📁 Project Structure

```text
Spam_SMS_Detection2/
│
├── data/
│ ├── raw/
│ │ └── spam.csv
│ ├── processed/
│ └── cleaned_data.csv
│
├── src/
│ ├── data_preprocessing.py
│ ├── utils.py
│ ├── model.py
│
├── app.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Spam_SMS_Detection2.git
cd Spam_SMS_Detection2

## ▶️ How to Run the Project
## Step 1: Data Preprocessing
python -m src.data_preprocessing
## Step 2: Train the Model
python src/model.py
## Step 3: Run Flask Web App
python app.py

###  Data Preprocessing Steps
Convert text to lowercase
Remove special characters and symbols
Remove numbers
Remove extra spaces
Clean and normalize text data
Encode labels (spam = 1, ham = 0)

### 🤖 Machine Learning Model
Algorithm: Naive Bayes (Best for text classification)
Feature Extraction: TF-IDF / CountVectorizer
Training Type: Supervised Learning


## 📊 Model Performance
```text 
Metric	Score
Accuracy	95%+
Precision	High
Recall	High

```

### 🌐 Web App Feature

The Flask web app allows users to:

Enter SMS text
Get instant prediction
View results in real-time


### 🚀 Future Improvements
🔥 Deep Learning models (LSTM, BERT)
🌍 Deploy on cloud platforms (Render / Heroku)
📱 Mobile app integration
📊 Advanced dashboard UI
🔗 API integration for real-time SMS filtering

## screenshots

<img width="621" height="449" alt="image" src="https://github.com/user-attachments/assets/ecfaa48a-f7a3-43dd-a976-bc9390291bca" />


<img width="956" height="431" alt="image" src="https://github.com/user-attachments/assets/41846f25-462a-4a37-82c6-b3424189a9e7" />


<img width="960" height="507" alt="image" src="https://github.com/user-attachments/assets/b8b480cf-65a7-4fea-b7db-5c8af0f17f5b" />



### 👨‍💻 Author
## Pratiksha Tomar


Machine Learning codsoft Internship Project


