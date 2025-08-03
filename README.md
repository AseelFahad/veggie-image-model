## 🥒🥕 Vegetable Image Classifier (Cucumber & Carrot)

A simple image classification project that uses a pre-trained Keras model to classify vegetable images into two categories: **cucumber** and **carrot**.

---

### 📁 Project Files

| File | Description |
|------|-------------|
| `keras_model.h5` | Pre-trained Keras model for classifying images |
| `labels.txt` | List of class labels (e.g., carrot, cucumber) |
| `predict.py` | Python script that loads and classifies an image |
| `Test.jpg`  | An image file used for testing |
| `Test2.jpg`  | An image file used for testing |


---

### 🔧 Requirements

Make sure the following libraries are installed:

```bash
pip install tensorflow pillow numpy
```

---

### 🚀 How to Use

1. Upload the following files to your working directory (e.g., in Google Colab or your local Python environment):
   - `keras_model.h5`
   - `labels.txt`
   - test images (e.g., `Test.jpg` , `Test2.jpg`)

2. Run the following script:

```python
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Prepare the image
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("Test2.jpg").convert("RGB")
image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
data[0] = normalized_image_array

# Run prediction
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Display the result
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
```

---

### ✅ Example Output

```
Class: Cucumber
Confidence Score: 0.9999387
---
Class: carrot
Confidence Score: 0.9378
```

---

### 📌 Notes

- The input image will be resized to **224x224** and converted to **RGB**.
- The model supports only two categories: **cucumber** and **carrot**.
- For best results, use clear images with only one vegetable per image.
