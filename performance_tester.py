from models import load_model_unet
from dataset import FlagOnGroundWithMask
import numpy as np
import time

# Load the UNet model
unet = load_model_unet('./checkpoints/unet.pth')
unet.eval()

# Load the dataset
dataset = FlagOnGroundWithMask('./data/flags', './data/desert')

# Set the number of samples to process
num_samples = 50  # Modify this number as needed

# Initialize a variable to track the total time
total_time = 0

# Process the specified number of samples
for i in range(num_samples):
    print(len(dataset))
    img, box = dataset[np.random.randint(0, len(dataset))]
    start = time.time()
    box_pred = unet(img.unsqueeze(0))
    end = time.time()

    # Calculate the time taken for this sample
    time_taken = end - start
    total_time += time_taken

    # Print the step and time taken
    print(f"Step {i + 1}/{num_samples}: Time taken: {time_taken:.6f} seconds, img shape: {img.shape}")

# Calculate and print the average time
average_time = total_time / min(num_samples, len(dataset))
print(f"Average time taken per sample: {average_time:.6f} seconds")
