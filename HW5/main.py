def main():
    print("Hello from hw5!")


if __name__ == "__main__":
    main()

    
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import requests
import pickle
with open('pipeline_v1.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    dict_vectorizer, model = pickle.load(f_in)

import pandas as pd
# input
data={
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X=dict_vectorizer.transform([data])
Y=model.predict_proba(X)[0,1]

print("Probability",Y)

