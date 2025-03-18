import pickle
import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained XGBoost model and feature list
with open("xgboost_model.pkl", "rb") as model_file:
    model_data = pickle.load(model_file)

xgb_clf = model_data["model"]
selected_features = model_data["features"]

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")

# Define the request model
class Transaction(BaseModel):
    V14: float
    V4: float
    V10: float
    V12: float
    V8: float
    V17: float
    V1: float
    V18: float
    V7: float
    V2: float
    V28: float
    V6: float
    V19: float
    Time: int  # Time will be converted to 'Hour'
    V11: float
    V27: float
    V3: float


@app.post("/predict")
async def predict(transaction: Transaction):
    transaction_dict = transaction.dict()

    # Convert Time into Hour feature
    transaction_dict["Hour"] = (transaction_dict["Time"] // 3600) % 24
    transaction_dict.pop("Time", None)  # Remove 'Time' after conversion

    # Ensure only selected features are used
    selected_features = ['V14', 'V4', 'V10', 'V12', 'V8', 'V17', 'V1',
                         'V18', 'V7', 'V2', 'V28', 'V6', 'V19', 'Hour', 
                         'V11', 'V27', 'V3']
    
    # Create DataFrame and ensure it contains only selected features
    
    input_df = pd.DataFrame([transaction_dict])
    input_df = input_df[selected_features]  # Select only required features

    # Perform prediction
    prediction = xgb_clf.predict(input_df)  
    probability = xgb_clf.predict_proba(input_df)[:, 1]  # Get probability of fraud class
    

    print("prediction",prediction)
    print("probability",probability)
    # Convert prediction to label
    fraud_label = "Fraud" if int(prediction) == 1 else "Non-Fraud"

    return {
        "fraud_prediction": fraud_label,  # Convert prediction to integer
        "fraud_probability": round(float(probability), 4)  # Ensure probability is a float
    }

# Run FastAPI (if running locally)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
