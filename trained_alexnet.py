import os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog

from alexnet import AlexNet

# Path to the saved checkpoint
checkpoint_path = 'tmp/finetune_alexnet/checkpoints/model_epoch10.ckpt'

# Initialize Tkinter root window (this will remain hidden)
root = Tk()
root.withdraw()

# Open a file dialog for image selection
image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

# Check if the user selected a file
if not image_path:
    print("No image selected. Exiting.")
    exit()

# Load the image and preprocess it
img = cv2.imread(image_path)
img = cv2.resize(img, (227, 227))
img = img.astype(np.float32)
img = np.expand_dims(img, axis=0)

# Define the label mapping
label_mapping = {
    0: 'neutral', 
    1: 'anger', 
    2: 'contempt', 
    3: 'disgust', 
    4: 'fear', 
    5: 'happy', 
    6: 'sadness', 
    7: 'surprise'
    }

# Placeholder for input image
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Placeholder for dropout rate
keep_prob = tf.placeholder(tf.float32)

# Initialize the AlexNet model
model = AlexNet(x, keep_prob, num_classes=7, skip_layer=['fc8'])

# Link variable to model output
score = model.fc8

# Softmax function to get probabilities
probs = tf.nn.softmax(score)

# Create a session to run the graph
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Create a saver object
    saver = tf.train.Saver()

    # Restore the model from the checkpoint
    saver.restore(sess, checkpoint_path)
    print("Model restored from:", checkpoint_path)

    # Make predictions
    feed_dict = {x: img, keep_prob: 1.0}  # No dropout during inference
    predictions = sess.run(probs, feed_dict=feed_dict)

    # Get the predicted label
    predicted_label = label_mapping[np.argmax(predictions)]

    print("Predicted Label:", predicted_label)
