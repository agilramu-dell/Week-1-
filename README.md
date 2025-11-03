# Week-1-
AI-Based Indian Cattle Breed Classifier
import RPi.GPIO as GPIO
import time
import Adafruit_DHT
import tensorflow as tf
import numpy as np
import cv2

# Sensor setup
DHT_SENSOR = Adafruit_DHT.DHT11
DHT_PIN = 4

# Load trained cattle breed model
model = tf.keras.models.load_model('cattle_breed_model.h5')
breed_labels = ['Tharparkar', 'Gir', 'Kankrej', 'Sahiwal', 'Ongole', ...]  # 50 breeds

# Initialize camera
camera = cv2.VideoCapture(0)

while True:
    # Capture image
    ret, frame = camera.read()
    img = cv2.resize(frame, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict breed
    predictions = model.predict(img)
    breed = breed_labels[np.argmax(predictions)]

    # Read environment data
    humidity, temperature = Adafruit_DHT.read_retry(DHT_SENSOR, DHT_PIN)

    print(f"Detected Breed: {breed}")
    print(f"Temperature: {temperature:.1f}°C, Humidity: {humidity:.1f}%")

    time.sleep(5)
