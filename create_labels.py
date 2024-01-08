import os

# Define the root directory where your image folders are located
root_dir = 'CK+48/'

# Define the labels and their corresponding folder names
labels = {'0': 'happy', 
          '1': 'sad', 
          '2': 'angry',
          '3': 'contempt',
          '4': 'disgust',
          '5': 'fear',
          '6': 'surprise'
          }

# Output text file to store image paths and labels
output_file = 'image_paths_labels.txt'

# Open the output file for writing
with open(output_file, 'w') as file:
    for label, folder_name in labels.items():
        folder_path = os.path.join(root_dir, folder_name)
        
        # Check if the folder exists
        if not os.path.exists(folder_path):
            continue
        
        # List all image files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

        # Write image paths and labels to the output file
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            file.write(f'{image_path} {label}\n')

print(f'Image paths and labels have been written to {output_file}')