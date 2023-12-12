import cv2
import numpy as np

# Read the image
image = cv2.imread('./identicard.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Iterate through the contours to find the largest rectangular contour (presumed to be the ID card)
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        # Found the rectangular contour
        x, y, w, h = cv2.boundingRect(contour)
        roi = image[y:y + h, x:x + w]

        # Mask the last digits by drawing a filled rectangle over them
        mask_color = (255, 255, 255)  # White color for the rectangle
        cv2.rectangle(roi, (w - 100, 0), (w, h), mask_color, thickness=cv2.FILLED)

        break

# Display the original and masked images
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', image)

# Save the masked image
cv2.imwrite('masked_image.jpg', image)

# Wait for a key press and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()