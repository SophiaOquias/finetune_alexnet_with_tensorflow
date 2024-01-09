# import os

# # Define the root directory where your image folders are located
# root_dir = 'CK+48/'

# # Define the labels and their corresponding folder names
# labels = {'0': 'happy', 
#           '1': 'sad', 
#           '2': 'anger',
#           '3': 'contempt',
#           '4': 'disgust',
#           '5': 'fear',
#           '6': 'surprise'
#           }

# # Output text file to store image paths and labels
# output_file = 'image_paths_labels.txt'

# # Open the output file for writing
# with open(output_file, 'w') as file:
#     for label, folder_name in labels.items():
#         folder_path = os.path.join(root_dir, folder_name)
        
#         # Check if the folder exists
#         if not os.path.exists(folder_path):
#             continue
        
#         # List all image files in the folder
#         image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

#         # Write image paths and labels to the output file
#         for image_file in image_files:
#             image_path = os.path.join(folder_path, image_file)
#             file.write(f'{image_path} {label}\n')

# print(f'Image paths and labels have been written to {output_file}')

import os
import random

# Path to the CK+ dataset
dataset_path = "CK+"

NEUTRAL = 0 

# Function to get emotion label from Emotion file
def get_emotion_label(emotion_file_path):
    with open(emotion_file_path, 'r') as file:
        label = float(file.readline().strip())
    return int(label)

# Create a list to store image paths and labels
image_paths_labels = []

# Iterate through image sequences
for root, dirs, files in os.walk(os.path.join(dataset_path, "cohn-kanade-images")):
    for file in files:
        if file.endswith(".png"):
            image_path = os.path.join(root, file)

            image_name = os.path.splitext(file)[0]

            # Extract subject and sequence information
            _, _, subject, sequence = root.split(os.sep)
            emotion_file_path = os.path.join(dataset_path, "Emotion", subject, sequence, f"{image_name}_emotion.txt")

            # Check if the emotion file exists
            if os.path.exists(emotion_file_path):
                label = get_emotion_label(emotion_file_path)
                image_paths_labels.append(f"{image_path} {label}")

                random_number = random.randint(0, 7)
                if(random_number == 0): 
                    neutral_path = os.path.join(dataset_path, "cohn-kanade-images", subject, sequence, f"{subject}_{sequence}_00000001.png")
                    image_paths_labels.append(f"{neutral_path} {NEUTRAL}")


# Write the image paths and labels to a text file
output_file_path = "image_paths_labels.txt"
with open(output_file_path, 'w') as output_file:
    for line in image_paths_labels:
        output_file.write(line + "\n")

print(f"File '{output_file_path}' created successfully.")

