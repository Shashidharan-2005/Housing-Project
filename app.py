from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = None
for f in os.listdir("artifacts"):
    if f.startswith("best_model_") and f.endswith(".joblib"):
        MODEL_PATH = os.path.join("artifacts", f)
        break

if MODEL_PATH and os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = model.named_steps['prep']
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
else:
    model, num_features, cat_features = None, [], []

@app.route("/", methods=["GET", "POST"])
def index():
    global model, num_features, cat_features
    prediction = None
    if not model:
        return "<h2>No trained model found. Please run train_fast.py first.</h2>"
    if request.method == "POST":
        input_data = {}
        for col in num_features:
            try:
                input_data[col] = float(request.form.get(col, 0))
            except:
                input_data[col] = 0.0
        for col in cat_features:
            input_data[col] = request.form.get(col, "")
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]
    return render_template("index.html", num_features=num_features, cat_features=cat_features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
