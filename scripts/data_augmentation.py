import random
import os
import cv2

import albumentations as A

path = "./dataset-frutas-original/DATASET_MANGO/PUNTOS NEGROS"
arr = os.listdir(path)

print('Procesando...')
for i in range(0, 208):
    img = cv2.imread(path+'/'+arr[i])
    transform = A.Compose([
        #A.HorizontalFlip(),
        #A.RandomBrightnessContrast(),
        #A.Rotate(p=1, limit=(0,360)),
        #A.OpticalDistortion(p=1, distort_limit=0.3, shift_limit=0.2),
        #A.Blur(p=1, blur_limit=25),
    ])
    random.seed(7)
    augmented_image = transform(image=img)['image']
    cv2.imwrite(path+'/aug_' + arr[i], augmented_image)

print('Terminado')
