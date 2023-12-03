
import csv
import os
import cv2
from tkinter import *
from tkinter.filedialog import askopenfilename

import FileDictIO

scale = 1

# generate visualizations of test data and corresponding model output
def show_pet_noses(test_file, output_file):

    test_noses = FileDictIO.file_to_dict(test_file)
    output_noses = FileDictIO.file_to_dict(output_file)

    for img in test_noses.keys():
        img_file = os.path.join('images/', img)
        if os.path.isfile(img_file):

            # loading and scaling image
            image = cv2.imread(img_file)
            dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
            imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

            # drawing circles for test data/prediction
            cv2.circle(imageScaled, test_noses[img], 8, (0, 255, 0), 1)
            cv2.circle(imageScaled, output_noses[img], 8, (255, 0,  0), 1)
            cv2.imshow(img_file, imageScaled)
            key = cv2.waitKey(0)
            cv2.destroyWindow(img_file)
            if key == ord('q'):
                exit(0)


if __name__ == '__main__':

    # choose input image folder
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    labels_file = askopenfilename(filetypes = [("Text files","*.txt")])
    path = os.path.dirname(labels_file)
    root.destroy()

    with open(labels_file, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row2 in reader:
            noseString = row2['nose']
            noseString = noseString[1:len(noseString) - 1]
            nose = tuple(map(int, noseString.split(',')))
            # imageFile = row2['image']
            noseImageFile = row2['image']
            print(noseImageFile, nose)
            imageFile = os.path.join(path, noseImageFile)
            if os.path.isfile(imageFile):
                image = cv2.imread(imageFile)
                dim = (int(image.shape[1] / scale), int(image.shape[0] / scale))
                imageScaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
                cv2.circle(imageScaled, nose, 2, (0, 0, 255), 1)
                cv2.circle(imageScaled, nose, 8, (0, 255, 0), 1)
                cv2.imshow(noseImageFile, imageScaled)
                key = cv2.waitKey(0)
                cv2.destroyWindow(noseImageFile)
                if key == ord('q'):
                    exit(0)



