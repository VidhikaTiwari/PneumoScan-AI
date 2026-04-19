from fastapi import FastAPI, File, UploadFile
from backend.app.schemas import PredictionResponse
from backend.app.inference import predict_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PneumoScan AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PneumoScan AI backend is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_image(image_bytes)
    return result