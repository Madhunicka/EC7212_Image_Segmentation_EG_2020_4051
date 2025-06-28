#----------------------------------------------------------
# Name: Madhunicka M.
# RegNo: EG/2020/4051
# Assignment 2
#----------------------------------------------------------

# required libraries
import numpy as np
import cv2
import os

def imageGenerate(width, height):
    image = np.zeros((height, width), dtype=np.uint8)
    image[:, :] = 255  # White background

    # Gray square
    square_size = width // 2
    square_x = 0 
    square_y = (height - square_size) // 2
    image[square_y:square_y+square_size, square_x:square_x+square_size] = 128  

    # Black circle
    cir_rad = width // 5
    cir_cent = (width - cir_rad, height // 2)
    cv2.circle(image, cir_cent, cir_rad, 0, -1) 

    return image

def addGaussiannoise(image):
    mean = 0
    stddev = 50
    img_float = image.astype(np.float32)
    noise = np.random.normal(mean, stddev, size=image.shape).astype(np.float32)
    img_noised = img_float + noise
    noisy_img = np.clip(img_noised, 0, 255).astype(np.uint8)
    return noisy_img


output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Generate and save original image
generatedImage = imageGenerate(300, 300)
cv2.imwrite(os.path.join(output_dir, "original_image.png"), generatedImage)

# Add noise and save
noisyImage = addGaussiannoise(generatedImage)
cv2.imwrite(os.path.join(output_dir, "noisy_image.png"), noisyImage)

# Apply Otsu's thresholding and save
_, otsuThreshold = cv2.threshold(noisyImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(output_dir, "otsu_threshold.png"), otsuThreshold)

# Optionally display images
cv2.imshow("Original", generatedImage)
cv2.imshow("Noisy", noisyImage)
cv2.imshow("Otsu Threshold", otsuThreshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
