import numpy as np 
import matplotlib.pyplot as plt


# specify the amount of images you want to see
IMGS = 1

# Read the npz file
for i in range(IMGS):
    data = np.load('RAVEN-10000\center_single\RAVEN_' + str(i) + '_train.npz')

    # Access the data in the npz file
    data_array = data['image']

    ## QUESTION
    # Create empty array to store the images
    image_grid = np.zeros((480, 480))

    # Create a loop to go through each image
    for j in range(8):
        # Extract each image from the dataset
        img = data['image'][j]
        # Reshape each image to (160, 160)
        img = img.reshape(160, 160)
        # Calculate the starting index of each image
        start_i = (j // 3) * 160
        start_j = (j % 3) * 160
        # Get the ending index of each image
        end_i = start_i + 160
        end_j = start_j + 160
        # Place the image in the empty array
        image_grid[start_i:end_i, start_j:end_j] = img

    # Plot the final image
    plt.title("Puzzle {}".format(i))
    plt.imshow(image_grid)
    plt.show()

    fig, axes = plt.subplots(1, 8, figsize=(20, 15))
    plt.title("Answers {}".format(i))
    for j in range(8):
        axes[j].imshow(data_array[-8+j, :, :])
    plt.show()

    image_grid[320:480, 320:480] = data_array[8+data['target']]
    plt.title('Solution {}'.format(i))
    plt.imshow(image_grid)
    plt.show()