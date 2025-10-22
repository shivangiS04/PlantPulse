import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from multimodal_fusion import multimodal_alert

# List all classes of your trained model
classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the fine-tuned MobileNetV2 model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load('best_plantpulse_mobilenetv2.pth', map_location=device))
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

st.title("PlantPulse: Multimodal Crop Stress Detection Demo")

uploaded_file = st.file_uploader("Upload a plant leaf image (JPG/PNG)", type=["jpg", "png"])
moisture = st.slider("Soil Moisture (%)", 0, 100, 50)
temp = st.slider("Ambient Temperature (°C)", 10, 50, 30)


def predict_image(image):
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    return {cls: prob for cls, prob in zip(classes, probs)}


if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    image_probs = predict_image(image)
    alert, stress_score = multimodal_alert(image_probs, moisture, temp)

    st.write("Prediction Probabilities:", image_probs)
    st.write(f"Alert Status: {'🚨 Crop Stress Detected!' if alert else '✅ Healthy Crop'}")
    st.write(f"Stress Score: {stress_score:.2f}")
