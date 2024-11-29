from models import load_model_retinanet
from data_preprocessing import FlagOnGroundDataset
import time

# Load the RetinaNet model
retinanet = load_model_retinanet('../checkpoints/retinanet_main.pth')
retinanet.eval()

# Load the dataset
dataset = FlagOnGroundDataset('../data/flags', '../data/desert')

# Set the number of samples to process
num_samples = 50  # Modify this number as needed

# Initialize a variable to track the total time
total_time = 0

# Process the specified number of samples
for i in range(min(num_samples, len(dataset))):
    img, box = dataset[i]
    start = time.time()
    box_pred = retinanet(img.unsqueeze(0))
    end = time.time()

    # Calculate the time taken for this sample
    time_taken = end - start
    total_time += time_taken

    # Print the step and time taken
    print(f"Step {i + 1}/{num_samples}: Time taken: {time_taken:.6f} seconds")

# Calculate and print the average time
average_time = total_time / min(num_samples, len(dataset))
print(f"Average time taken per sample: {average_time:.6f} seconds")
