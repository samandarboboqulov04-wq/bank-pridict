# Bank Customer Prediction Web App

This is the web version of the bank customer prediction model. It includes the same ML model from the desktop app, but now runs as an interactive web application.

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web App

To start the web application, run:

```bash
streamlit run app_web.py
```

The app will automatically open in your default browser at `http://localhost:8501`

## Features

- **Interactive Input Form**: Enter customer information using intuitive input fields
- **Instant Predictions**: Get model predictions in real-time
- **Responsive Design**: Works on desktop and mobile browsers
- **Professional UI**: Clean, modern interface with professional styling

## Input Fields

- Customer Age
- Number of Products
- Months with Bank
- Number of Products Held
- Inactive Months (12 months)
- Contact Count (12 months)
- Credit Limit
- Credit Balance
- Credit Portion (%)
- Total Transaction Amount

## Model

The web app uses the pre-trained ML model saved in `model (1).pkl`. This model can predict customer churn and other banking metrics.

## Stopping the App

Press `CTRL+C` in the terminal to stop the Streamlit server.
