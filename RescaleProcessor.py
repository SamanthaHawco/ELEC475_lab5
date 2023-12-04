# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import cv2
import FileDictIO


# downscales input dictionary (original size) labels to newly scaled AxA image (128x128 in our case)
def original_to_scaled(input_dict, new_size=(128, 128)):
    scaled_dict = {}
    for key in input_dict.keys():

        # calculating scaling factor
        img = cv2.imread('images/' + key)
        img_dim = (img.shape[1], img.shape[0])
        scale_factors = rescale(img_dim, new_size)

        # scale labels
        labels = input_dict[key]
        scaled_label = (round(labels[0] / scale_factors[0]), round(labels[1] / scale_factors[1]))

        # append to new scaled dict
        scaled_dict[key] = scaled_label

    return scaled_dict


# upscales input dictionary (AxA) labels to original image size (NxM) image
def scaled_to_original(input_file, old_size=(128, 128)):

    input_dict = FileDictIO.file_to_dict(input_file)

    scaled_dict = {}
    for key in input_dict.keys():

        # calculating scaling factor
        img = cv2.imread('images/' + key)
        img_dim = (img.shape[1], img.shape[0])
        scale_factors = rescale(old_size, img_dim)

        # scale labels
        labels = input_dict[key]
        scaled_label = (round(labels[0] / scale_factors[0]), round(labels[1] / scale_factors[1]))

        # append to new scaled dict
        scaled_dict[key] = scaled_label

    return scaled_dict


def rescale(dim, new_size):  # function for calculating scaling factor for (x,y) of image labels
    x_scale = dim[0] / new_size[0]
    y_scale = dim[1] / new_size[1]
    return x_scale, y_scale
