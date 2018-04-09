"""
For this project we use the Yale Face Database. It contains 165 images in total.
The data is of 15 persons with 11 images per person.
The title of each image is in the format “Subject99.expression”
where the expression can be sad, happy, surprised etc.

It also contains images with different configurations
such as center light, right light, with glasses etc.
We train our classifier using all the images except those which have a “sad” expression.
Those with “sad” expression would be used for testing our classifier.
Each subject has a number which acts as a label for that subject.
Although this label can also be a name but for easier data representation we take it as number.
"""

import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import cv2

CASCADELOCATION = os.path.normpath(
    os.path.realpath(cv2.__file__) +
    "/../data/haarcascade_frontalface_default.xml")
FACECASCADE = cv2.CascadeClassifier(CASCADELOCATION)


def prepare_dataset(directory):
    """Prepare image dataset"""
    paths = [
        os.path.join(directory, filename) for filename in os.listdir(directory)
        if "sad" not in filename
    ]
    images = []
    labels = []
    row = 140
    col = 140
    for path in paths:
        image_pil = Image.open(path).convert('L')
        image = np.array(image_pil, 'uint8')
        nbr = int(os.path.split(path)[1].split('.')[0].replace("subject", ""))
        faces = FACECASCADE.detectMultiScale(image)
        for (x, y, w, h) in faces:
            images.append(image[y:y + col, x:x + row])
            labels.append(nbr)
            cv2.imshow("Reading Faces ", image[y:y + col, x:x + row])
            cv2.waitKey(50)
    return images, labels, row, col


DIRECTORY = './yalefaces'
IMAGES, LABELS, ROW, COL = prepare_dataset(DIRECTORY)
N_COMPONENTS = 10
cv2.destroyAllWindows()
PCA = PCA(n_components=N_COMPONENTS, whiten=True, svd_solver='randomized')

PARAM_GRID = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
}
CLF = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), PARAM_GRID)

TESTING_DATA = []
for i, _ in enumerate(IMAGES):
    TESTING_DATA.append(IMAGES[i].flatten())
PCA = PCA.fit(TESTING_DATA)
TRANSFORMED = PCA.transform(TESTING_DATA)
CLF.fit(TRANSFORMED, LABELS)

IMAGE_PATHS = [
    os.path.join(DIRECTORY, filename) for filename in os.listdir(DIRECTORY)
    if filename.endswith('sad')
]
for image_path in IMAGE_PATHS:
    pred_image_pil = Image.open(image_path).convert('L')
    pred_image = np.array(pred_image_pil, 'uint8')
    pred_faces = FACECASCADE.detectMultiScale(pred_image)
    for (x, y, w, h) in pred_faces:
        X_test = PCA.transform(
            np.array(pred_image[y:y + COL, x:x + ROW]).reshape(1, -1))
        mynbr = CLF.predict(X_test)
        nbr_act = int(
            os.path.split(image_path)[1].split('.')[0].replace("subject", ""))
        print("Predicted By Classifier : ", mynbr[0], " Actual : ", nbr_act)
