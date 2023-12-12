import cv2
import numpy as np
import pytesseract
from PIL import Image

def find_and_mask_numbers(image):
  cv2.imshow('test_imge', image)


image_path = './identicard.jpg'
find_and_mask_numbers(image_path)