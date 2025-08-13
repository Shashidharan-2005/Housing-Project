import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def run(train_csv, test_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    target = "SalePrice"
    X = train.drop(columns=[target])
    y = train[target]
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)

    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))]), cat_cols)
    ])

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.001, max_iter=2000),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=120, max_depth=None),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=150, learning_rate=0.1, max_depth=3),
    }

    results = []
    best_name, best_model, best_r2 = None, None, -1e9
    for name, est in models.items():
        pipe = Pipeline([("prep", preprocess), ("model", est)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xva)
        rmse = mean_squared_error(yva, pred, squared=False)
        r2 = r2_score(yva, pred)
        results.append({"model": name, "MSE": rmse**2, "RMSE": rmse, "R2": r2})
        if r2 > best_r2:
            best_name, best_model, best_r2 = name, pipe, r2

    pd.DataFrame(results).sort_values("R2", ascending=False).to_csv(os.path.join(out_dir, "holdout_results.csv"), index=False)
    best_model.fit(X, y)
    joblib.dump(best_model, os.path.join(out_dir, f"best_model_{best_name}.joblib"))
    preds = best_model.predict(test)
    pd.DataFrame({"Id": test["Id"] if "Id" in test.columns else range(len(test)), "SalePrice": preds}).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    print(f"Best model: {best_name}, R2: {best_r2}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    run(a.train, a.test, a.out)
