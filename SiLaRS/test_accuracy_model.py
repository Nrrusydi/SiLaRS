import os
import numpy as np
from tensorflow.keras.models import load_model

# Set the path to your test dataset
test_dir = 'path/to/test/dataset'

# Load the saved model
model = load_model('sign_language_model.h5')

# Set the parameters for evaluation
batch_size = 16

# Load and preprocess the test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate_generator(
    test_generator,
    steps=test_generator.n // batch_size
)

print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
