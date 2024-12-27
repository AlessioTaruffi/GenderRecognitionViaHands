import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

# Definisci la classe per la trasformazione personalizzata
class CustomAlexNetTransform:
    def __call__(self, pil_image):
        # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        # Optional: normalizing to [0..1] before blur
        np_image_norm = imageNormalization(np_image)

        # Blur
        blurred = cv2.GaussianBlur(np_image_norm, (7, 7), 0)

        # Resize to 224×224
        resized = cv2.resize(blurred, (224, 224))

        # Convert back to 0..255
        final_8u = restoreOriginalPixelValue(resized)  # shape: (224, 224, 3)

        # Return PIL image (mode='RGB')
        return Image.fromarray(final_8u, mode='RGB')


class CustomLeNetTransform:
    def __call__(self, pil_image):
         # Convert PIL -> RGB -> NumPy
        pil_image = pil_image.convert('RGB')
        np_image = np.array(pil_image, dtype=np.uint8)

        # Convert to GRAY correctly
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        # Equalize hist
        contrast = cv2.equalizeHist(gray_image)

        # Normalize [0..255] -> [0..1]
        norm = imageNormalization(contrast)

        # Resize to 32×32
        resized = cv2.resize(norm, (32, 32))

        # Convert back to [0..255] uint8
        final_8u = restoreOriginalPixelValue(resized)  # shape: (32, 32)

        # Return PIL image (mode='L' = single channel)
        return Image.fromarray(final_8u, mode='L')


# To normalize one image [values range 0:1]
def imageNormalization(image: np.ndarray):
    # E.g., convert from [0..255] to [0..1] float
    return image.astype(np.float32) / 255.0

# To restore the original pixel scale -> cast on int 
def restoreOriginalPixelValue(image: np.ndarray):
    # Convert from [0..1] float back to [0..255] uint8
    return (image * 255).astype(np.uint8)

'''
# To pre process a single image 
def preProcessingData(image: np.ndarray, type: str):
    if type == "palmar":
         return preProcessingPalm(image)
    elif type == "dorsal":
        return preProcessingDorsal(image)
    else:
        return None

# To pre process palm image
def preProcessingPalm(image: np.ndarray):
    # Convert the RGB image in GRAYSCALE image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # Enhancing the constrast to add details 
    contrastImage = cv2.equalizeHist(grayImage)
    # Normalized the image
    normalizedImage = imageNormalization(contrastImage)
    # Image resize for LeNet standard size
    resizedImage = cv2.resize(normalizedImage, (224,224))
    # Convert the grayscale image back to RGB
    rgbImage = cv2.cvtColor(resizedImage, cv2.COLOR_GRAY2RGB)
    # Restore the original image pixel value -> NECESSARIO ORA? 
    finalImage = restoreOriginalPixelValue(rgbImage)
    return finalImage

# To pre process dorsal image
def preProcessingDorsal(image: np.ndarray):
    imageNormalized = imageNormalization(image)
    ''''''
    Blur the image using cv2.GaussianBlur(image, (3, 3), 0) where (3,3) rappresent the kernel dimension and 0 is the standard deviation for gaussian distribution along the x axis
    cv2.GaussianBlur applica, ad ogni pixel, una sfocatura gaussiana utilizzando una funzione gaussiana per calcolare i valori medi dei pixel vicini.
    Effettua una stima pesando andando a dare maggior peso ai pixel più vicini rispetto a quelli più lontani
    SigmaX = 0 nella nostra funzione in modo che OpenCV calcola automaticamente la deviazione standard basandosi sulla dimensione del kernel
    ''''''
    blurredImage = cv2.GaussianBlur(imageNormalized, (7, 7), 0) 
    # Image resize for AlexNet standard size
    resizedImage = cv2.resize(blurredImage, (224,224))
    # Restore the original image pixel value -> NECESSARIO ORA? 
    finalImage = restoreOriginalPixelValue(resizedImage)
    return finalImage
'''

# Build AlexNet trasformations
def buildAlexNetTransformations():
    return transforms.Compose([
        CustomAlexNetTransform(),
        transforms.ToTensor(),          # Converte le immagini in tensori
    ])

# Build LeNet trasformations
def buildLeNetTransformations():
    return transforms.Compose([
        CustomLeNetTransform(),
        transforms.ToTensor(),          # Converte le immagini in tensori
    ])