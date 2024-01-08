from datagenerator import ImageDataGenerator

# Define the path to the class list text file
class_list_file = 'train.txt'

# Instantiate the ImageDataGenerator

batch_size = 32  # Set your desired batch size
data_generator = ImageDataGenerator(txt_file=class_list_file, mode='training', batch_size=batch_size, num_classes=7)
images, labels = data_generator.next_batch()

print(len(labels))