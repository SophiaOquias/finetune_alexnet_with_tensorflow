# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# from tkinter import Tk, filedialog

# from alexnet import AlexNet

# # Path to the saved checkpoint
# checkpoint_path = 'tmp/finetune_alexnet/checkpoints/model_epoch48.ckpt'

# # Initialize Tkinter root window (this will remain hidden)
# root = Tk()
# root.withdraw()

# # Open a file dialog for image selection
# image_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

# # Check if the user selected a file
# if not image_path:
#     print("No image selected. Exiting.")
#     exit()

# # Load the image and preprocess it
# img = cv2.imread(image_path)
# img = cv2.resize(img, (227, 227))
# img = img.astype(np.float32)
# img = np.expand_dims(img, axis=0)

# # Define the label mapping
# label_mapping = {
#     0: 'neutral', 
#     1: 'anger', 
#     2: 'contempt', 
#     3: 'disgust', 
#     4: 'fear', 
#     5: 'happy', 
#     6: 'sadness', 
#     7: 'surprise'
#     }

# # Placeholder for input image
# x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# # Placeholder for dropout rate
# keep_prob = tf.placeholder(tf.float32)

# # Initialize the AlexNet model
# model = AlexNet(x, keep_prob, num_classes=8, skip_layer=None)

# # Link variable to model output
# score = model.fc8

# # Softmax function to get probabilities
# probs = tf.nn.softmax(score)

# # Create a session to run the graph
# with tf.Session() as sess:
#     # Initialize variables
#     sess.run(tf.global_variables_initializer())

#     # Create a saver object
#     saver = tf.train.Saver()

#     # Restore the model from the checkpoint
#     saver.restore(sess, checkpoint_path)
#     print("Model restored from:", checkpoint_path)

#     # Make predictions
#     feed_dict = {x: img, keep_prob: 1.0}  # No dropout during inference
#     predictions = sess.run(probs, feed_dict=feed_dict)

#     # Get the predicted label
#     predicted_label = label_mapping[np.argmax(predictions)]

#     print("Predicted Label:", predicted_label)

import os
import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, filedialog
import warnings

from alexnet import AlexNet  # Assuming you have the AlexNet model implemented in 'alexnet.py'

# Suppress NumPy FutureWarnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# Path to the saved checkpoint
checkpoint_path = 'tmp/finetune_alexnet/checkpoints/model_epoch48.ckpt'

# Read image paths and labels from "testframes.txt"
with open('testframes.txt', 'r') as file:
    lines = file.readlines()

# Extract image paths and labels
image_paths = [line.split()[0] for line in lines]
ground_truth_labels = [int(line.split()[1]) for line in lines]

# Initialize Tkinter root window (this will remain hidden)
root = Tk()
root.withdraw()

# Initialize lists to store predictions and ground truth labels
predictions_list = []

# Placeholder for input image
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Placeholder for dropout rate
keep_prob = tf.placeholder(tf.float32)

# Initialize the AlexNet model
model = AlexNet(x, keep_prob, num_classes=8, skip_layer=None)

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

    # Make predictions for each image
    for image_path, ground_truth_label in zip(image_paths, ground_truth_labels):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (227, 227))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        # Make predictions
        feed_dict = {x: img, keep_prob: 1.0}  # No dropout during inference
        predictions = sess.run(probs, feed_dict=feed_dict)

        # Get the predicted label
        predicted_label = np.argmax(predictions)
        predictions_list.append(predicted_label)

# Calculate overall accuracy
correct_predictions = np.sum(np.array(predictions_list) == np.array(ground_truth_labels))
total_images = len(ground_truth_labels)
overall_accuracy = correct_predictions / total_images * 100

print(f"Overall Accuracy: {overall_accuracy}%")

label_mapping = {
    0: 'Neutral', 
    1: 'Anger', 
    2: 'Contempt', 
    3: 'Disgust', 
    4: 'Fear', 
    5: 'Happy', 
    6: 'Sadness', 
    7: 'Surprise'
    }

# Calculate and print accuracy per class
unique_labels = set(ground_truth_labels)
for label in unique_labels:
    if label in ground_truth_labels:
        correct_predictions_class = np.sum(np.array(predictions_list)[np.array(ground_truth_labels) == label] == label)
        total_images_class = np.sum(np.array(ground_truth_labels) == label)
        accuracy_class = correct_predictions_class / total_images_class * 100
        print(f"Class {label_mapping[label]} Accuracy: {accuracy_class}%")

