from PIL import Image
import numpy as np
import pytesseract

filename = './assets/identicard.jpg'
img = np.array(Image.open(filename))
text = pytesseract.image_to_string(img)
print(text)