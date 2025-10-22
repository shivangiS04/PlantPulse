import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from multimodal_fusion import multimodal_alert
from gradcam_utils import GradCAM
import numpy as np
import cv2
import random

classes = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load('best_plantpulse_mobilenetv2.pth', map_location=device))
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Crop stress facts for user engagement
facts = [
    "Over 70% of crop loss occurs due to plant stress and disease.",
    "Maintaining soil moisture above 50% greatly reduces water stress.",
    "Temperature above 30°C can impact photosynthesis and crop health.",
    "Crop rotation and fungicides help manage late blight in potatoes.",
    "Healthy leaves typically appear green and firm with no spots or curling."
]

remedies = {
    True: "Recommendation: Increase watering, control temperature and monitor disease prevention.",
    False: "Your crop looks healthy. Maintain current care and monitor regularly."
}

# Display app title
st.title("PlantPulse: Multimodal Crop Stress Detection Demo")

# Show random fact
st.info(facts[random.randint(0, len(facts)-1)])

# File uploader and sliders
uploaded_file = st.file_uploader("Upload a plant leaf image (JPG/PNG)", type=["jpg", "png"])
moisture = st.slider("Soil Moisture (%)", 0, 100, 50, help="Simulated sensor reading: 0 (dry) to 100% (wet)")
temp = st.slider("Ambient Temperature (°C)", 10, 50, 30, help="Simulated sensor reading for temperature")

# Sidebar help and FAQ
st.sidebar.header("Help & FAQ")
st.sidebar.info("""
- **What is this app?**
    - AI system detecting crop stress from images and sensor data.
- **How to use?**
    - Upload a leaf image, adjust moisture and temperature values.
- **What is crop stress?**
    - Signs of disease, nutrient deficiency or environmental hazards.
- **What is Grad-CAM?**
    - Visual explanation showing regions important for AI predictions.
""")

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
    st.success(remedies[alert])
    st.write(f"Stress Score: {stress_score:.2f}")

    if st.button("Show Grad-CAM Explanation"):
        img_t = transform(image).unsqueeze(0).to(device)
        gradcam = GradCAM(model, model.features[-1])
        cam = gradcam(img_t, class_idx=np.argmax(list(image_probs.values())))
        img_np = np.array(image.resize((224, 224)))
        cam_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_np, 0.6, cam_colored, 0.4, 0)
        st.image(overlay, caption="Grad-CAM Explanation")

# Footer
st.markdown("""
---
Made with 💡 by **Shivangi Singh**
""", unsafe_allow_html=True)
