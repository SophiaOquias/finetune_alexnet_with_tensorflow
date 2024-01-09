import tensorflow as tf
import numpy as np
from alexnet import AlexNet

# Function to load weights from the checkpoint file
def load_alexnet_weights(session, checkpoint_path):
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)

# Input image placeholder
x = tf.placeholder(tf.float32, [None, 227, 227, 3])

# Dropout placeholder
keep_prob = tf.placeholder(tf.float32)

# Number of classes
num_classes = 7  # Change this according to your needs

# Create an instance of AlexNet
alexnet = AlexNet(x, keep_prob, num_classes, skip_layer=['fc8'])

# Create TensorFlow session
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())

    # Load pretrained weights
    load_alexnet_weights(sess, 'tmp/finetune_alexnet/checkpoints\model_epoch10.ckpt')

    # Read and resize the input image using TensorFlow functions
    image_path = 'CK+48/happy/happy_23.png'  # Use forward slash for paths
    image_string = tf.read_file(image_path)
    input_image = tf.image.decode_png(image_string, channels=3)  # Use decode_png
    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)  # Convert to float32
    input_image = tf.image.resize_images(input_image, [227, 227])
    input_image = tf.subtract(input_image, [123.68, 116.779, 103.939])  # Subtract mean
    input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension

    # Run the session to get the prediction
    input_image_np = sess.run(input_image)

    # Now you can use this session to classify the input image
    prediction = sess.run(alexnet.fc8, feed_dict={x: input_image_np, keep_prob: 1.0})
    print(prediction)


    # Convert prediction to label
    predicted_label = np.argmax(prediction)
    print(predicted_label)
    label_mapping = {0: 'happy', 1: 'sad', 2: 'anger', 3: 'contempt', 4: 'disgust', 5: 'fear', 6: 'surprise'}
    predicted_class = label_mapping[predicted_label]

    print(f"The predicted class is: {predicted_class}")
