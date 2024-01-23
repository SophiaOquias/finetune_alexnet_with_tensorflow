import csv
import os
import random

# Mapping of numeric labels to emotions
labels_mapping = {
    'Neutral': '0',
    'Happy': '1',
    'Sad': '2',
    'Anger': '3',
    'Contempt': '4',
    'Disgust': '5',
    'Fear': '6',
    'Surprise': '7'
}

def convert_time_to_ms(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

def extract_frames(root_folder, file_name, labels_file, timestamp_column='timestamp'):
    # Initialize an empty list to store the results
    results = []

    # Read labels csv
    with open(labels_file, 'r') as labels_csv:
        labels_reader = csv.DictReader(labels_csv)
        for row in labels_reader:
            start_time_ms = convert_time_to_ms(row['Start Time'])
            end_time_ms = convert_time_to_ms(row['End Time'])

            # Read frames csv
            frames_file = os.path.join(root_folder, f"{file_name}.csv")
            with open(frames_file, 'r') as frames_csv:
                frames_reader = csv.DictReader(frames_csv)
                for frame_row in frames_reader:
                    # Strip leading and trailing spaces from all column names
                    frame_row = {key.strip(): value.strip() for key, value in frame_row.items()}
                    
                    if timestamp_column in frame_row:
                        random_integer = random.randint(1, 30)
                        if random_integer % 10 == 0: 
                            timestamp_ms = float(frame_row[timestamp_column])
                            if start_time_ms <= timestamp_ms <= end_time_ms:
                                image_path = os.path.join(root_folder, f"{file_name}_aligned/frame_det_00_{frame_row['frame'].zfill(6)}.png")

                                numeric_label = row['Classes']
                                emotion_label = labels_mapping.get(numeric_label, 'unknown')
                                results.append((image_path, emotion_label))

    # Write results to the output file after processing all rows
    write_to_output(results)

def write_to_output(results):
    output_file = 'testframes.txt'
    with open(output_file, 'a') as output_txt:
        for image_path, label in results:
            output_txt.write(f"{image_path} {label}\n")

if __name__ == "__main__":
    for i in range(1, 11): 
        participant_number = i
        root_folder = f"recordings/{participant_number}_1"
        file_name = f"{participant_number}_1"
        labels_file = os.path.join(root_folder, f"{file_name}_labels.csv")

        # Check if output file exists, remove if it does
        output_file = 'testframes.txt'

        extract_frames(root_folder, file_name, labels_file)
        print(f"Script executed successfully. Output written to {output_file}")
