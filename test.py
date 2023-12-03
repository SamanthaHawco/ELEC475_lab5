# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import argparse
import math
import os

import numpy
import torch
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, transforms
from torch.utils.data import DataLoader

import FileDictIO
import RescaleProcessor
import model as net
import show_pet_noses
from PetDataset import PetDataset

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-cuda', type=str, help='[y/N]')
parser.add_argument('-b', "--batch_size", type=int, default=8, help='batch size for data loaders')
parser.add_argument('-params', "--params_file", type=str, help='trained model file')
args = parser.parse_args()

# device
device = torch.device('cpu')
if (args.cuda == 'y' or args.cuda == 'Y') and torch.cuda.is_available():
    device = torch.device('cuda')

# set output directory
out_dir = './output/'
os.makedirs(out_dir, exist_ok=True)

# model
trained_params = torch.load(args.params_file)
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = net.PetNet(resnet)
model.fc_layers.load_state_dict(trained_params)
print('model loaded OK!')

# Data Loaders
transform = Compose([
    ToTensor(),
    transforms.Resize((128, 128), antialias=True)
])

# testing data
testset = PetDataset(img_dir='images', training=False, transform=transform)
test_loader = DataLoader(dataset=testset, batch_size=int(args.batch_size), shuffle=False)


def test():
    model.eval()
    model.to(device)

    # initialize evaluation variables
    total_imgs = 0
    correct_pred = 0
    threshold = 0.50  # this can change, to be more or less strict
    distances = []

    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for imgs, (x_labels, y_labels) in test_loader:

            # combine x and y labels
            labels = torch.stack([x_labels, y_labels], dim=1).float()

            imgs = imgs.squeeze().to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)

            total_imgs += labels.size(0)

            # calculate Euclidean distance for each point
            for i in range(0, labels.size(0)):
                x1 = labels[i].data[0].item()
                y1 = labels[i].data[1].item()
                x2 = outputs[i].data[0].item()
                y2 = outputs[i].data[1].item()

                all_outputs.append((x2, y2))

                # threshold calc

                if (x1 * (1 + threshold) > x2 > x1 * (1 - threshold)) and (
                        y1 * (1 + threshold) > y2 > y1 * (1 - threshold)):
                    correct_pred += 1

                e_dist = math.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2))  # eucliden distance
                distances += [e_dist]

    # generate .txts for visualization
    l_output_dict = {}
    o_output_dict = {}
    for i in range(0, labels.size(0)):
        # Label = (x1, y1)
        x1 = int(labels[i].data[0].item())
        y1 = int(labels[i].data[1].item())

        # Output = (x2, y2)
        x2 = int(outputs[i].data[0].item())
        y2 = int(outputs[i].data[1].item())

        # get image that is going to be visualized
        img = FileDictIO.get_image_from_coord((x1, y1), 'downscaled_test_noses.txt')

        # generate dict entry and add to dict
        l_output_dict[img] = (x1, y1)
        o_output_dict[img] = (x2, y2)

    # write subset of labels to file
    FileDictIO.dict_to_file(l_output_dict, 'downscaled_visualizations_test.txt')
    FileDictIO.dict_to_file(o_output_dict, 'downscaled_visualizations_output.txt')

    # upscale visualizations (and create visualization file)
    scaled_dict = RescaleProcessor.scaled_to_original('downscaled_visualizations_test.txt')
    FileDictIO.dict_to_file(scaled_dict, 'visualizations_test.txt')
    scaled_dict = RescaleProcessor.scaled_to_original('downscaled_visualizations_output.txt')
    FileDictIO.dict_to_file(scaled_dict, 'visualizations_output.txt')

    # calculate generic accuracy
    accuracy = round((correct_pred / total_imgs) * 100, 2)
    print(f'General Accuracy: {accuracy}% for threshold: {threshold}')

    # calculate accuracy statistics
    min_dist = round(min(distances), 2)
    mean_dist = round(numpy.mean(distances), 2)
    max_dist = round(max(distances), 2)
    std_dev_dist = round(numpy.std(distances), 2)

    # print accuracy statistics 
    print(f'Min Distance: {min_dist}px')
    print(f'Mean Distance: {mean_dist}px')
    print(f'Max Distance: {max_dist}px')
    print(f'Standard Deviation of Distance: {std_dev_dist}px')

    # show visualizations
    show_pet_noses.show_pet_noses('visualizations_test.txt', 'visualizations_output.txt')

if __name__ == "__main__":
    test()
