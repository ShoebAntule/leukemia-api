from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
models_dict = {}
model_dir = os.path.join(os.path.dirname(__file__), '../models')
try:
    models_dict["cnn"] = tf.saved_model.load(os.path.join(model_dir, '1'))
except Exception as e:
    logging.warning("Failed to load CNN model: {}".format(e))
    models_dict["cnn"] = None

try:
    gru_path = os.path.join(os.path.dirname(__file__), '../Training/models/3')
    if os.path.exists(gru_path):
        models_dict["gru"] = tf.saved_model.load(gru_path)
        logging.info("CNN+GRU model loaded successfully")
    else:
        logging.warning("CNN+GRU model path does not exist")
        models_dict["gru"] = None
except Exception as e:
    logging.warning("Failed to load CNN+GRU model: {}".format(e))
    models_dict["gru"] = None

CLASS_NAMES = ['basophil', 'erythroblast', 'monocyte', 'myeloblast', 'seg_neutrophil']


@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):
    model_name = request.query_params.get('model', 'cnn')
    # Parse model name to handle versions like 'cnn:1'
    base_model_name = model_name.split(':')[0]
    print("Query params:", dict(request.query_params))
    print("Model name:", model_name)
    print("Base model name:", base_model_name)
    print("MODEL in dict:", models_dict.get(base_model_name))
    if base_model_name not in models_dict or models_dict[base_model_name] is None:
        print("Raising 400")
        raise HTTPException(status_code=400, detail="Invalid or unavailable model")
    MODEL = models_dict[base_model_name]
    print("MODEL assigned:", MODEL)

    data = await file.read()
    print("Data length:", len(data))
    image = read_file_as_image(data)
    image_info = {
        'original_dimensions': "{}x{}".format(image.shape[1], image.shape[0]),
        'file_size_kb': round(len(data) / 1024, 2),
        'processed_resolution': "264x264"
    }
    image = tf.image.resize(image, (264, 264))
    img_batch = np.expand_dims(image, 0)
    img_batch = tf.cast(img_batch, tf.float32) / 255.0  # Normalize to match training

    infer = MODEL.signatures['serving_default']
    predictions = infer(inputs=img_batch)['output_0']
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    logging.info("Prediction with {}: {}, confidence: {}".format(model_name, predicted_class, confidence))

    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'image_info': image_info
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000)
