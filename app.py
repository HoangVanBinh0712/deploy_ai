# %%
import os
import uvicorn
from fastapi import FastAPI
import numpy as np
from tensorflow import keras
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# %%
app = FastAPI()
# Load model trained
filterCV = keras.models.load_model('models_P')
# create global parameter
max_length = 800
trunc_type = 'post'
pad_type = 'post'


# Load the lable
encoding_to_label_in = open("./dictionary.pickle", "rb")
encoding_to_label = pickle.load(encoding_to_label_in)
# load feature_tokenizer to transfer data to number array
feature_tokenizer_in = open("./feature_tokenizer.pickle", "rb")
feature_tokenizer = pickle.load(feature_tokenizer_in)

class predictBody(BaseModel):
    resume: str
    skill: str


def transform(data: str):
    clean_data = clean_text(data)
    data_sequence = feature_tokenizer.texts_to_sequences([clean_data])
    data_padded = pad_sequences(
        data_sequence, maxlen=max_length, padding=pad_type, truncating=trunc_type)

    return np.array(data_padded)


@app.post('/predict')
def predict_cv(predictBody: predictBody):

    resume_padded = transform(predictBody.resume)
    skill_padded = transform(predictBody.skill)
    # predict
    prediction = filterCV.predict((resume_padded, skill_padded))

    # Get top 5 highest %
    indices = np.argpartition(prediction[0], -5)[-5:]
    indices = indices[np.argsort(prediction[0][indices])]
    indices = list(reversed(indices))

    # Concat data to return
    result_data = []
    for index in indices:
        result_data.append({str(encoding_to_label[index]): str(
            round(prediction[0][index]*100, 2)) + "% "})

    return JSONResponse(content=jsonable_encoder({"results": result_data}))


@app.get('')
def get_home():
    return {'message': 'Wellcome'}


def clean_text(resume_text):
    try:
        resume_text = re.sub('http\S+\s*', ' ', resume_text)
        resume_text = re.sub('RT|cc', ' ', resume_text)
        resume_text = re.sub('#\S+', '', resume_text)
        resume_text = re.sub('@\S+', '  ', resume_text)
        resume_text = re.sub('[%s]' % re.escape(
            """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)
        resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
        resume_text = re.sub('\s+', ' ', resume_text)
        resume_text = resume_text.lower()
        return resume_text
    except:
        return ''


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ.get("PORT", 5000)))

# %%
