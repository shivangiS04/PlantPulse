def normalize_sensor(moisture, temp):
    """Normalize sensor data into a stress score (0 to 1)."""
    moisture_score = max(0, (50 - moisture) / 50)  # More stress if moisture < 50%
    temp_score = max(0, (temp - 30) / 20)  # More stress if temp > 30°C
    return (moisture_score + temp_score) / 2


def multimodal_alert(image_probs, moisture, temp, threshold=0.5):
    """
    Fuse image classification probabilities with sensor data to generate crop stress alert.

    Args:
        image_probs (dict): Class probabilities from image model (e.g., {'Tomato_healthy': 0.8, ...})
        moisture (float): Soil moisture percentage [0-100]
        temp (float): Ambient temperature in Celsius
        threshold (float): Threshold above which alert is triggered

    Returns:
        alert (bool): True if crop stress detected
        stress_score (float): Combined stress score value (0-1)
    """
    healthy_class_name = 'Tomato_healthy'  # Adjust if your class name varies
    healthy_prob = image_probs.get(healthy_class_name, 0)
    sensor_score = normalize_sensor(moisture, temp)

    # Combine image and sensor scores into a final stress score
    stress_score = (1 - healthy_prob + sensor_score) / 2
    alert = stress_score > threshold
    return alert, stress_score
