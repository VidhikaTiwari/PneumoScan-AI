# PneumoScan AI

AI-powered chest X-ray screening platform that classifies scans as **Normal** or **Pneumonia** using deep learning. Built as a full-stack machine learning application with a deployed React frontend and FastAPI backend.


## Features

- Upload chest X-ray images for analysis
- AI prediction: **Normal** vs **Pneumonia**
- Confidence score for each prediction
- Class probability breakdown
- Low-confidence warning system
- Clean multi-page frontend experience
- Fully deployed full-stack application

---

## Tech Stack

### Frontend
- React
- Vite
- CSS
- Axios
- React Router

### Backend
- FastAPI
- Python
- Uvicorn

### Machine Learning
- PyTorch
- torchvision
- ResNet18 (Transfer Learning)
- scikit-learn

### Deployment
- Vercel (Frontend)
- Render (Backend)

---

## Model Performance

Evaluated on the test dataset:

- Accuracy: **87.82%**
- Precision: **84.43%**
- Recall: **98.72%**
- F1 Score: **91.02%**

---

## Live Demo

Frontend: https://pneumo-scan-ai.vercel.app  
Backend API Docs: https://pneumoscan-ai-6zfz.onrender.com/docs

---