import cv2
import numpy as np

# To normalize one image 
def imageNormalization(image: np.ndarray):
    return np.float32(image)/255

# To pre process a single image 
def prepocessingData(image: np.ndarray, type: str):
    if type == "palmar":
         return prepocessingPalm(image)
    elif type == "dorsal":
        return prepocessingPalm(image)
    else:
        return None

# To pre process palm image
def prepocessingPalm(image: np.ndarray):
    # Convert the RGB image in GRAYSCALE image
    normalizedGrayImage = cv2.cvtColor(imageNormalization(image), cv2.COLOR_BGR2GRAY)
    # Enhancing the constrast to add details 

    #cv2.equalizeHist(normalizedGrayImage)
    pass

# To pre process dorsal image
def prepocessingDorsal(image: np.ndarray):
    imageNormalized = imageNormalization(image)
    pass

