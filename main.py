
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from models.BankNote import BankNotes
import numpy as np
import pickle 
import pandas as pd

app = FastAPI()

pickle_in = open('models/model.pkl', "rb")
classifier = pickle.load(pickle_in)


# This is simple a homepage route
@app.get('/')
async def index():
    return {'message': 'Welcome to Bank Note Classification app'}

@app.post('/predict')
async def predict_banknote(data: BankNotes):
    data = data.dict()
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    if prediction[0] >= 0.5:
        prediction = "Fake Note"
    else:
        prediction = "It's a Bank Note"
    return {
        'prediction': prediction
    }
    

if __name__ == "__main__":

    uvicorn.run(app, host = '127.0.0.1', port = 8000)