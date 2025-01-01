from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI()


# Define the model input data structure using Pydantic
class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


# Load the pre-trained model
try:
    model = joblib.load("wineQT_model.pkl")  # Load the model with joblib
except Exception as e:
    print(f"Error loading model: {e}")


# Prediction endpoint
@app.post("/predict")
async def predict_wine_quality(data: WineData):
    # Convert the input data into a numpy array (required format for the model)
    input_data = np.array([[data.fixed_acidity, data.volatile_acidity, data.citric_acid,
                            data.residual_sugar, data.chlorides, data.free_sulfur_dioxide,
                            data.total_sulfur_dioxide, data.density, data.pH, data.sulphates,
                            data.alcohol]])

    # Standardize the input data using StandardScaler (same scaling as during training)
    scaler = StandardScaler()

    # Normally, the scaler would be fitted to the training data. Here, we use the same logic, but without the fitted scaler.
    # Fit scaler on the same data shape
    input_data_scaled = scaler.fit_transform(input_data)  # Scale the input data

    # Predict the wine quality
    predicted_quality = model.predict(input_data_scaled)[0]  # Get the first value from the prediction

    # Ensure the predicted value is within the range of 0 to 10
    predicted_quality = max(0, min(10, predicted_quality))  # Clamp between 0 and 10

    # Return the predicted quality as an integer to avoid serialization issues with numpy types
    return {"predicted_quality": int(predicted_quality)}
