from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS
import os

# Create the Flask app first
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
CORS(app)  # Enable CORS

# Load ML model and encoders
model = joblib.load("persona_model.pkl")
scaler = joblib.load("scaler.pkl")
education_encoder = joblib.load("education_encoder.pkl")
occupation_encoder = joblib.load("occupation_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_persona():
    try:
        data = request.get_json()

        # Create DataFrame with proper columns
        input_df = pd.DataFrame([{
            "Age": data["age"],
            "Occupation": data["occupation"],
            "Experience": data["experience"],
            "Education": data["education"]
        }])

        # Encode categorical features
        input_df["Education"] = education_encoder.transform(input_df["Education"])
        input_df["Occupation"] = occupation_encoder.transform(input_df["Occupation"])

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict persona
        prediction = model.predict(scaled_input)[0]
        persona = label_encoder.inverse_transform([prediction])[0]

        return jsonify({"persona": persona})

    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 400


# Run the app (only if this file is run directly)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
