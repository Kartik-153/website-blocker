import tensorflow as tf
import numpy as np
from selenium import webdriver
import hashlib
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

class ProductivityModel:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.model.compile()

    def predict_productivity(self, image_path: str) -> bool:
        # Load and preprocess the image
        img = tf.keras.utils.load_img(image_path, target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict productivity using the model
        prediction = self.model.predict(img_array)
        return bool(prediction[0][0] < 0.5)

    def take_screenshot(self, url: str) -> str:
        # Initialize the Chrome WebDriver
        driver = webdriver.Chrome()
        driver.get(url)
        
        # Wait until the body element is present
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Hash the URL to create a unique filename
        hashed_url = hashlib.sha256(url.encode()).hexdigest()
        image_path = "tensor/dataset/prod/" + hashed_url + ".png"
        
        # Save the screenshot and quit the driver
        driver.save_screenshot(image_path)
        driver.quit()
        
        return image_path