#!/usr/bin/env python3
# main.py
"""
FastAPI service that predicts likely Pokémon teammates for a
Regulation I restricted core.  Optimised for low-memory deployment on
Render (≤ 512 MiB).
"""
from __future__ import annotations

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import uvicorn

# ───────────────────────────────────────────────
# Paths
# ───────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATAFRAMES_DIR  = os.path.join(BASE_DIR, "dataframes")

MODEL_FILE      = os.path.join(MODELS_DIR, "vgc_regi_restrictedcore_model.joblib")
X_DF_FILE       = os.path.join(DATAFRAMES_DIR, "X_df.csv")

# ───────────────────────────────────────────────
# Globals (populated in `startup`)
# ───────────────────────────────────────────────
model           = None          # scikit-learn multi-output classifier
label_columns   = None          # List[str] – order must match model
X_df            = None          # pd.DataFrame with correct columns only

# ───────────────────────────────────────────────
# FastAPI app
# ───────────────────────────────────────────────
app = FastAPI(title="Pokémon VGC Teammate Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ───────────────────────────────────────────────
# Fallbacks and sprite helper
# ───────────────────────────────────────────────
fallbacks: dict[str, str] = {
    "ogerpon-cornerstone": "ogerpon-cornerstone-mask",
    "ogerpon-hearthflame": "ogerpon-hearthflame-mask",
    "ogerpon-wellspring":  "ogerpon-wellspring-mask",
    "ogerpon":             "ogerpon-teal-mask",
    "landorus":            "landorus-incarnate",
    "tornadus":            "tornadus-incarnate",
    "thundurus":           "thundurus-incarnate",
    "enamorus":            "enamorus-incarnate",
    "urshifu":             "urshifu-single-strike",
    "indeedee-f":          "indeedee-female",
    "giratina":            "giratina-altered",
}

def get_sprite_url(poke_name: str) -> str:
    """
    Returns a sprite URL for `poke_name` using PokéAPI.
    Prefers official-artwork PNG (transparent background).
    Falls back to the transparent 0.png sprite if none found.
    """
    base = poke_name.lower().replace(" ", "-").replace("’", "").replace("'", "")
    attempts: List[str] = []
    if base in fallbacks:
        attempts.append(fallbacks[base])
    attempts.extend([base, base.split("-")[0]])

    for attempt in attempts:
        try:
            res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{attempt}", timeout=4)
            res.raise_for_status()
            data = res.json()
            # Prefer official-artwork (transparent), then front_default (classic)
            sprite = (
                data["sprites"]["other"]["official-artwork"]["front_default"]
                or data["sprites"]["front_default"]
            )
            if sprite:
                return sprite
        except Exception:
            continue

    return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png"

# ───────────────────────────────────────────────
# Prediction helper
# ───────────────────────────────────────────────
def predict_teammates(core: Tuple[str, str],
                      top_n: int = 20) -> pd.DataFrame:
    """
    core : tuple of two Pokémon names
    Returns a DataFrame with columns ['Teammate', 'Predicted Probability']
    """
    if model is None or X_df is None or label_columns is None:
        raise RuntimeError("Model and data not loaded")

    input_row = pd.DataFrame(
        0,
        index=[0],
        columns=X_df.columns,
        dtype=np.int8,          # tiny dtype, saves memory
    )

    for mon in core:
        col = f"core_{mon}"
        if col in input_row.columns:
            input_row.at[0, col] = 1
        else:
            print(f"[WARN] Unknown core feature: {col}", flush=True)

    # model.predict_proba returns one array per label for OneVsRest setup
    probs: List[float] = []
    for est, prob_arr in zip(model.estimators_, model.predict_proba(input_row)):
        # prob_arr shape (n_samples, 2) for binary problems
        if prob_arr.shape[1] == 2:
            probs.append(prob_arr[0, 1])
        else:  # rare edge case
            label_idx = est.classes_[1] if 1 in est.classes_ else est.classes_[0]
            probs.append(1.0 if label_idx == 1 else 0.0)

    teammate_names = [c.replace("teammate_", "") for c in label_columns]
    ranked = sorted(zip(teammate_names, probs),
                key=lambda x: x[1],
                reverse=True)

    # Filter out probability <= 0
    ranked = [row for row in ranked if row[1] > 0]

    # Only keep up to top_n
    ranked = ranked[:top_n]

    return pd.DataFrame(ranked,
                        columns=["Teammate", "Predicted Probability"])

# ───────────────────────────────────────────────
# Request / response models
# ───────────────────────────────────────────────
class TeammateRequest(BaseModel):
    core1: str
    core2: str

# ───────────────────────────────────────────────
# API routes
# ───────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Pokémon Team Builder API is running!"}

@app.post("/predict-teammates")
async def predict_teammates_endpoint(req: TeammateRequest):
    try:
        results = predict_teammates((req.core1, req.core2))
        results["sprite_url"] = results["Teammate"].apply(get_sprite_url)
        return results.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ───────────────────────────────────────────────
# Startup: load model & dataframe lazily
# ───────────────────────────────────────────────
@app.on_event("startup")
def load_assets() -> None:
    global model, label_columns, X_df

    print("[DEBUG] startup: begin loading assets", flush=True)

    # --- Load model bundle lazily ---
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] Model file not found: {MODEL_FILE}", flush=True)
        sys.exit(1)

    bundle = load(MODEL_FILE, mmap_mode="r")   # mmap keeps RAM usage low
    model          = bundle["model"]
    label_columns  = bundle["label_columns"]

    # --- Only read header of X_df.csv ---
    if not os.path.exists(X_DF_FILE):
        print(f"[ERROR] X_df.csv not found: {X_DF_FILE}", flush=True)
        sys.exit(1)

    X_df = pd.read_csv(X_DF_FILE, nrows=0)
    print("[DEBUG] startup: finished loading assets", flush=True)

# ───────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────
if __name__ == "__main__":
    # Render supplies $PORT (default 10000).  Fall back to 8000 locally.
    port_env = os.environ.get("PORT")
    print(f"[DEBUG] Render supplied PORT={port_env}", flush=True)

    port = int(port_env or 8000)
    print(f"[DEBUG] Starting Uvicorn on port {port}", flush=True)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
    )