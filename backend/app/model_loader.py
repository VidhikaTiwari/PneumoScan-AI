import json
import torch
import torch.nn as nn
from torchvision import models

ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/best_model.pth"
CLASS_NAMES_PATH = f"{ARTIFACTS_DIR}/class_names.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()