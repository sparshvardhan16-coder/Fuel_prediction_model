# Fuel Efficiency Prediction Model

This is a Flask-based web application that predicts the fuel efficiency of vehicles across different ethanol blends (E5, E10, and E20). It uses a pre-trained machine learning model (Support Vector Regression - SVR) to estimate these metrics based on various vehicle engine parameters.

## Features
- **Interactive Web Interface:** User-friendly web form to easily input vehicle details.
- **Accurate Predictions:** Uses a fine-tuned SVR model to ensure reliable results.
- **Multi-Blend Support:** Predicts fuel efficiency for E5, E10, and E20 ethanol fuel blends simultaneously.
- **Dynamic Encoding:** Safely encodes categorical data (`Vehicle Name`, `Engine Type`) into numerical inputs seamlessly.

## Inputs Requested
To get a prediction, the model requires the following vehicle parameters:
- **Vehicle Name** (e.g., selected from a predefined list of vehicles)
- **Engine Type** (e.g., Petrol, Diesel, Hybrid)
- **Engine Capacity (L)**
- **Number of Cylinders**
- **Displacement** (cc)
- **Acceleration**

## Project Structure
- `app_.py`: The main Flask application script housing the backend and routes.
- `requirements.txt`: Listed Python package dependencies required to run the environment.
- `ml_test.ipynb`: Jupyter Notebook containing data exploration, model training, and testing details.
- `dataset/`: Contains the base dataset (`india_vehicle_fuel_efficiency_updated.csv`) used to train the models.
- `artifacts/`: Stores the serialized machine learning models (`.pkl` files) and encoders.
    - `best_svr_tuned.pkl`: The primary predictive model used by the web app.
    - `label_encoders.pkl`: Fitted encoders for turning categorical variables into numeric values.
- `catboost_info/`: Logs and details tied to CatBoost testing during model development.
- `templates/`: HTML templates for the frontend UI.
    - `index.html`: The main web page.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sparshvardhan16-coder/Fuel_prediction_model.git
   cd Fuel_prediction_model
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app_.py
   ```

5. **View in Browser:**
   Open your preferred web browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```
