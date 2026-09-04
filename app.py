from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)
model = joblib.load("model.pkl")

FLOWER_NAMES = ["Setosa", "Versicolor", "Virginica"]

FLOWER_DETAILS = {
    "Setosa": {
        "scientific": "Iris setosa", "family": "Iridaceae", "genus": "Iris",
        "species": "I. setosa", "color": "Purple / Blue-violet",
        "shape": "Small and distinctive flower", "sepal": "Short and broad",
        "petal": "Short and narrow", "habitat": "Wet meadows, marshes and stream banks",
        "distribution": "Northern regions of North America", "class_name": "Monocotyledons",
        "common_name": "Setosa Iris",
        "description": "Iris setosa is one of the three species represented in the classic Iris dataset. It generally has smaller petals compared with the other two classes.",
        "image": "setosa.jpg"
    },
    "Versicolor": {
        "scientific": "Iris versicolor", "family": "Iridaceae", "genus": "Iris",
        "species": "I. versicolor", "color": "Blue / Violet",
        "shape": "Showy iris flower", "sepal": "Medium length", "petal": "Medium length",
        "habitat": "Wetlands, marshes and moist areas", "distribution": "North America",
        "class_name": "Monocotyledons", "common_name": "Northern Blue Flag",
        "description": "Iris versicolor is commonly known as the northern blue flag and is associated with wet and moist habitats.",
        "image": "versicolor.jpg"
    },
    "Virginica": {
        "scientific": "Iris virginica", "family": "Iridaceae", "genus": "Iris",
        "species": "I. virginica", "color": "Blue / Violet",
        "shape": "Large iris flower", "sepal": "Long and broad", "petal": "Longer and wider",
        "habitat": "Wetlands and moist habitats", "distribution": "Eastern and central North America",
        "class_name": "Monocotyledons", "common_name": "Virginia Iris",
        "description": "Iris virginica is a wetland iris characterized by attractive blue-violet flowers.",
        "image": "virginica.jpg"
    }
}

def validate_measurements(sl, sw, pl, pw):
    if not 4.0 <= sl <= 8.0: return "Sepal Length must be between 4.0 and 8.0 cm."
    if not 2.0 <= sw <= 4.5: return "Sepal Width must be between 2.0 and 4.5 cm."
    if not 1.0 <= pl <= 7.0: return "Petal Length must be between 1.0 and 7.0 cm."
    if not 0.1 <= pw <= 2.5: return "Petal Width must be between 0.1 and 2.5 cm."
    return None

def predict_values(sl, sw, pl, pw):
    sample = np.array([[sl, sw, pl, pw]])
    result = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]
    # Supports both integer and string class outputs.
    if isinstance(result, (int, np.integer)):
        flower = FLOWER_NAMES[int(result)]
    else:
        flower = str(result).title()
    confidence = round(float(max(probs)) * 100, 2)
    probabilities = {FLOWER_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(len(probs))}
    return flower, confidence, probabilities

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = confidence = probabilities = details = error = None
    values = {"sepal_length":"","sepal_width":"","petal_length":"","petal_width":""}
    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"]); sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"]); pw = float(request.form["petal_width"])
            values = {"sepal_length":sl,"sepal_width":sw,"petal_length":pl,"petal_width":pw}
            error = validate_measurements(sl, sw, pl, pw)
            if not error:
                prediction, confidence, probabilities = predict_values(sl, sw, pl, pw)
                details = FLOWER_DETAILS[prediction]
        except ValueError:
            error = "Please enter valid numerical values."
        except Exception as e:
            error = "Prediction error: " + str(e)
    return render_template("index.html", prediction=prediction, confidence=confidence,
                           probabilities=probabilities, details=details, error=error, values=values)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        sl=float(data["sepal_length"]); sw=float(data["sepal_width"])
        pl=float(data["petal_length"]); pw=float(data["petal_width"])
        error=validate_measurements(sl,sw,pl,pw)
        if error: return jsonify({"success":False,"error":error}), 400
        flower, confidence, probabilities = predict_values(sl,sw,pl,pw)
        return jsonify({"success":True,"flower":flower,"confidence":confidence,
                        "probabilities":probabilities,
                        "image":"/static/images/"+FLOWER_DETAILS[flower]["image"],
                        "scientific_name":FLOWER_DETAILS[flower]["scientific"],
                        "time":datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")})
    except Exception as e:
        return jsonify({"success":False,"error":str(e)}), 400

@app.route("/health")
def health():
    return jsonify({"status":"online","application":"Iris Flower Classifier",
                    "model":"Random Forest","framework":"Flask",
                    "developer":"Rashpreet Kaur Arora","classes":FLOWER_NAMES})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
          
