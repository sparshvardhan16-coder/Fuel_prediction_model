from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# load trained model + encoders
model = joblib.load("artifacts/best_svr_tuned.pkl")
encoders = joblib.load("artifacts/label_encoders.pkl")

feature_cols = [
    'Vehicle Name',
    'Engine Capacity (L)',
    'Engine Type',
    'no_of_cylinders',
    'displacement',
    'acceleration'
]

def safe_encode(le, value):
    if value in le.classes_:
        return le.transform([value])[0]
    return -1

@app.route("/", methods=["GET", "POST"])
def index():

    # defaults (for first load)
    vehicle_name = None
    engine_type = None
    engine_capacity = ""
    no_of_cylinders = ""
    displacement = ""
    acceleration = ""

    prediction = None
    
    if request.method == "POST":
        vehicle_name = request.form.get("vehicle_name")
        engine_type = request.form.get("engine_type")
        engine_capacity = request.form.get("engine_capacity")
        no_of_cylinders = request.form.get("no_of_cylinders")
        displacement = request.form.get("displacement")
        acceleration = request.form.get("acceleration")

        # convert numerics
        engine_capacity = float(engine_capacity)
        no_of_cylinders = float(no_of_cylinders)
        displacement = float(displacement)
        acceleration = float(acceleration)

        # encoding
        vehicle_encoded = safe_encode(encoders["Vehicle Name"], vehicle_name)
        engine_encoded = safe_encode(encoders["Engine Type"], engine_type)

        X = pd.DataFrame([[
            vehicle_encoded,
            engine_capacity,
            engine_encoded,
            no_of_cylinders,
            displacement,
            acceleration
        ]], columns=feature_cols)

        pred = model.predict(X)[0]
        prediction = {
            "E5": round(pred[0],2),
            "E10": round(pred[1],2),
            "E20": round(pred[2],2)
        }

    return render_template(
        "index.html",
        prediction=prediction,
        vehicle_list=list(encoders["Vehicle Name"].classes_),
        engine_list=list(encoders["Engine Type"].classes_),

        selected_vehicle=vehicle_name,
        selected_engine=engine_type,
        selected_engine_capacity=engine_capacity,
        selected_cylinders=no_of_cylinders,
        selected_displacement=displacement,
        selected_acceleration=acceleration
    )

if __name__ == "__main__":
    app.run(debug=True)
