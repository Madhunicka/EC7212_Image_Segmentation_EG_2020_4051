#----------------------------------------------------------
# Name: Madhunicka M.
# RegNo: EG/2020/4051
# Assignment 2
#----------------------------------------------------------

# required libraries
import cv2
import numpy as np
import os

def show_segmentation(mask):
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)

def region_growing(image, seed_points, threshold_range):
    mask = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=np.uint8)

    height, width = image.shape
    queue = list(seed_points)
    seed_intensities = [image[pt[1], pt[0]] for pt in seed_points]

    iteration = 0

    while queue:
        iteration += 1
        x, y = queue.pop(0)

        if visited[y, x]:
            continue

        visited[y, x] = 1
        pixel_value = image[y, x]

        if any(abs(int(pixel_value) - int(seed_val)) <= threshold_range for seed_val in seed_intensities):
            mask[y, x] = 255

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                        queue.append((nx, ny))

        if iteration % 500 == 0:
            show_segmentation(mask)

    return mask

input_path = "input/img.jpg"
image_color = cv2.imread(input_path)

if image_color is None:
    raise FileNotFoundError("Could not load the image.")

# Convert to grayscale
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Define seed point
# Example seed points (x, y) coordinates in the image
seed_points = [(240, 180), (260, 240), (180, 160)]  

# Threshold range for intensity similarity
threshold_range = 15

# Run region-growing
segmented = region_growing(image_gray, seed_points, threshold_range)

# Save output
output_path = "results/segmented_mask.png"
cv2.imwrite(output_path, segmented)
print(f"✅ Segmentation mask saved to: {output_path}")

# Display
cv2.imshow("Original Image", image_color)
cv2.imshow("Segmented Mask", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()
