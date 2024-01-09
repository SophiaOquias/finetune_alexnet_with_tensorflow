from sklearn.model_selection import train_test_split

# Load the class list (image paths and labels) from your text file
class_list_file = 'image_paths_labels.txt'

# Read the class list into lists of image paths and labels
with open(class_list_file, 'r') as file:
    lines = file.readlines()
    image_paths = [line.split()[0] for line in lines]
    labels = [line.split()[1] for line in lines]

# Split the data into training and validation sets (e.g., 80% training, 20% validation)
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.3, random_state=42)

# Define output file paths for training and validation sets
train_output_file = 'train.txt'
val_output_file = 'val.txt'

# Write training data to the training output file
with open(train_output_file, 'w') as train_file:
    for path, label in zip(train_paths, train_labels):
        train_file.write(f'{path} {label}\n')

# Write validation data to the validation output file
with open(val_output_file, 'w') as val_file:
    for path, label in zip(val_paths, val_labels):
        val_file.write(f'{path} {label}\n')
