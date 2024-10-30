# Deepfake Detection with TensorFlow

This repository contains code for detecting deepfakes using a convolutional neural network (CNN) built with TensorFlow and Keras. The model is based on the Xception architecture and is trained to classify frames extracted from videos.

## Prerequisites

Make sure you have the following installed:

- Python 3.x
- TensorFlow
- OpenCV
- Keras

You can install the required packages using pip:

```bash
pip install tensorflow opencv-python keras


Getting Started
Clone the Repository

Clone the FaceForensics repository, which contains the dataset for training:

bash
Copy code
git clone https://github.com/ondyari/FaceForensics.git
Extract Video Frames

Use the video_to_frames function to convert videos into frames:

python
Copy code
import cv2

def video_to_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame{count}.jpg", frame)
        count += 1
    cap.release()
Call this function with your video path and desired output folder.

Preprocess Frames

Preprocess the frames before feeding them into the model:

python
Copy code
import cv2
import numpy as np

def preprocess_frame(frame, img_size=224):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame / 255.0  # Normalize to [0, 1]
    return frame
Build and Compile the Model

The model is built using the Xception architecture:

python
Copy code
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Prepare Data for Training

Use ImageDataGenerator to create training and validation data generators:

python
Copy code
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory('/content/FaceForensics', target_size=(224, 224), batch_size=32, class_mode='binary', subset='training')
val_gen = train_datagen.flow_from_directory('/content/FaceForensics', target_size=(224, 224), batch_size=32, class_mode='binary', subset='validation')
Train the Model

Fit the model using the training data:

python
Copy code
model.fit(train_gen, epochs=10, validation_data=val_gen)
Evaluate the Model

Evaluate the model’s performance on the validation dataset:

python
Copy code
loss, accuracy = model.evaluate(val_gen)
print(f"Validation Accuracy: {accuracy}")
Save the Model

Save the trained model for future use:

python
Copy code
model.save('deepfake_detection_model.h5')
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
FaceForensics Dataset
sql
Copy code

### Customization
- Feel free to modify the content, especially the sections about installation, usage, and any specific notes about the dataset.
- You might also want to add sections for troubleshooting, known issues, or future improvements 
