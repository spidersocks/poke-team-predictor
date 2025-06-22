# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from joblib import load  # Import load from joblib
import uvicorn  # Import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Paths ===

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATAFRAMES_DIR = os.path.join(BASE_DIR, "dataframes")

# === Load Model and Data ===
try:
    # Construct the full path to the model file
    model_path = os.path.join(MODELS_DIR, "vgc_regi_restrictedcore_model.joblib")

    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

     bundle = load(model_path, mmap_mode='r')
    model = bundle["model"]
    label_columns = bundle["label_columns"]

    # Construct the full path to the X_df.csv file
    X_df_path = os.path.join(DATAFRAMES_DIR, "X_df.csv")

    # Check if the X_df.csv file exists
    if not os.path.exists(X_df_path):
        raise FileNotFoundError(f"X_df.csv file not found at: {X_df_path}")

    X_df = pd.read_csv(X_df_path, nrows=0)

    print("Model and data loaded successfully.")

except FileNotFoundError as e:
    print(f"Error loading model or data: {e}")
    model = None  # Set model to None to indicate loading failure
    label_columns = None
    X_df = None
except Exception as e:
    print(f"Unexpected error during model/data loading: {e}")
    model = None
    label_columns = None
    X_df = None

# === Helper Functions ===

fallbacks = {
    "ogerpon-cornerstone": "ogerpon-cornerstone-mask",
    "ogerpon-hearthflame": "ogerpon-hearthflame-mask",
    "ogerpon-wellspring": "ogerpon-wellspring-mask",
    "ogerpon": "ogerpon-teal-mask",
    "landorus": "landorus-incarnate",
    "tornadus": "tornadus-incarnate",
    "thundurus": "thundurus-incarnate",
    "enamorus": "enamorus-incarnate",
    "urshifu": "urshifu-single-strike",
    "indeedee-f": "indeedee-female",
    "giratina": "giratina-altered"
}

import requests

def get_sprite_url(pokemon_name):
    base_name = pokemon_name.lower().replace(" ", "-").replace("’", "").replace("'", "")
    name_attempts = []
    if base_name in fallbacks:
        name_attempts.append(fallbacks[base_name])
    name_attempts.append(base_name)
    name_attempts.append(base_name.split("-")[0])

    for attempt in name_attempts:
        try:
            res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{attempt}")
            res.raise_for_status()
            data = res.json()

            sprite_url = (
                data["sprites"]["front_default"]
                or data["sprites"]["other"]["official-artwork"]["front_default"]
            )

            if sprite_url:
                return sprite_url
        except Exception:
            continue

    # Return transparent fallback instead of None
    return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png"

def predict_teammates(core, model, X_df, label_columns, top_n=20):
    core = list(core)
    input_row = pd.DataFrame(columns=X_df.columns)
    input_row.loc[0] = 0

    for mon in core:
        col = f"core_{mon}"
        if col in input_row.columns:
            input_row.at[0, col] = 1
        else:
            print(f"Warning: {col} not in input features")

    probs = []
    for i, prob_arr in enumerate(model.predict_proba(input_row)):
        if prob_arr.shape[1] == 2:
            probs.append(prob_arr[0, 1])
        else:
            label_idx = model.estimators_[i].classes_[0]
            probs.append(1.0 if label_idx == 1 else 0.0)

    teammate_names = [col.replace("teammate_", "") for col in label_columns]
    results = sorted(zip(teammate_names, probs), key=lambda x: x[1], reverse=True)[:top_n]
    return pd.DataFrame(results, columns=["Teammate", "Predicted Probability"])

# === Request Model ===

class TeammateRequest(BaseModel):
    core1: str
    core2: str

# === API Endpoints ===

@app.get("/")
def read_root():
    return {"message": "Pokemon Team Builder API is running!"}

@app.post("/predict-teammates")
async def predict_teammates_endpoint(req: TeammateRequest):
    global model, X_df, label_columns  # Access the global variables

    if model is None or X_df is None or label_columns is None:
        raise HTTPException(status_code=500, detail="Model or data failed to load.")

    try:
        core_pair = (req.core1, req.core2)
        results = predict_teammates(core_pair, model, X_df, label_columns)
        results['sprite_url'] = results['Teammate'].apply(get_sprite_url)  # Add sprite URLs
        results_list = results.to_dict(orient="records")
        return results_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)