# BERT-Based Enhanced Intention Recognition for Large-Scale Banking Customer Queries

## Project Overview

This project implements an AI-powered banking query classification system using a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model. The system automatically identifies customer intentions from banking-related queries and classifies them into predefined banking service categories, enabling intelligent query routing and improved customer support.

## Features

- BERT-based Intent Recognition
- Banking Query Classification
- Real-Time Intent Prediction
- Confidence Score Generation
- Secure User Authentication
- User Registration and Login
- Session Management
- SQLite Database Integration
- Flask-Based Web Application
- Prediction History Tracking
- Context-Aware Natural Language Processing
- Scalable Banking Support Solution

## Technologies Used

- Python
- Flask
- PyTorch
- Hugging Face Transformers
- BERT (bert-base-uncased)
- SQLite
- HTML
- CSS
- JavaScript

## Project Structure

```
Banking-Intent-Recognition-System/
│
├── app.py
├── requirements.txt
├── intension.ipynb
├── README.md
│
├── models/
│   ├── config.json
│   ├── label_encoder.pkl
│   ├── label_mappings.pkl
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│
└── templates/
    ├── login.html
    ├── signup.html
    └── prediction.html
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Banking-Intent-Recognition-System.git
cd Banking-Intent-Recognition-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

## System Modules

### User Interaction Module
Provides a web interface for user registration, login, query submission, and result visualization.

### Authentication Module
Handles user registration, login validation, password security, and session management.

### Intent Recognition Module
Processes banking queries using a fine-tuned BERT transformer model and predicts the most relevant banking intent.

### Database Management Module
Stores user information, query history, prediction results, and confidence scores using SQLite.

## Sample Banking Queries

- Check my account balance
- Transfer money to another account
- Block my debit card
- Track my loan application
- Request bank statement
- Add a new beneficiary
- Credit card payment issue
- ATM cash not received

## Future Enhancements

- Multilingual Banking Query Support
- Voice-Based Query Processing
- Cloud Deployment
- Real-Time Model Retraining
- Advanced Analytics Dashboard
- Integration with Banking APIs

## Author

**S. Saanthosh**  
Master of Computer Applications (MCA)  
SRM Institute of Science and Technology

## License

This project is developed for academic and educational purposes.
