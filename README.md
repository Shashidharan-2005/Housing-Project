# Surprise Housing Price Prediction (Flask Version)

## Steps to use
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (this will create artifacts/best_model_*.joblib):
   ```bash
   python src/train_fast.py --train "Housing-project-train-data.csv" --test "Hosuing-project-test-data.csv" --out artifacts
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your browser at `http://127.0.0.1:5000`.

If you skip step 2, the app will say "No trained model found."
