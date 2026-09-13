import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Load the final dataset
df = pd.read_csv("final_sih_master_training_dataset.csv")

# 2. Define the target variable (continuous stress percentage)
y = df["overall_stress_score"]

# 3. Drop only the actual non-predictive/target columns to isolate features (X)
X = df.drop(columns=["timestamp", "target_diagnosis", "overall_stress_score"])

# 4. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Initialize and train the Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = rf_model.predict(X_test)
print(f"\nMean Absolute Error (MAE): {mean_absolute_error(y_test, predictions):.2f}")
print(f"R-Squared (R2) Score: {r2_score(y_test, predictions):.4f}")

# 7. Extract Feature Importances
importance_df = pd.DataFrame({
    "Feature": X.columns, 
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("\n--- TOP 5 STRESS DRIVERS ---")
print(importance_df.head(5))

# 8. Export the trained model for deployment
joblib.dump(rf_model, "sih_stress_rf_model.pkl")