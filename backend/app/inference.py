from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms

from backend.app.model_loader import model, class_names, DEVICE

IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.70

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image_bytes: bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        predicted_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_index].item()

        probability_dict = {
            class_names[i]: round(probabilities[0][i].item(), 4)
            for i in range(len(class_names))
        }

    warning = None
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warning = "Low confidence prediction. Review manually."

    return {
        "predicted_class": class_names[predicted_index],
        "confidence": round(confidence, 4),
        "probabilities": probability_dict,
        "warning": warning
    }