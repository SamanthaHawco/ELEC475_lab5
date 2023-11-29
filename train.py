# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import argparse
import datetime
import os
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader as DataLoader
import model as net
from PetDataset import PetDataset

# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', "--epochs", type=int, default=10, help='number of epochs for training')
parser.add_argument('-b', "--batch_size", type=int, default=8, help='batch size for data loaders')
parser.add_argument('-s', "--fc_output", type=str, help='output model file for frontend')
parser.add_argument('-p', "--loss_plot", type=str, help='loss plot file')
parser.add_argument('-cuda', type=str, help='[y/N]')
parser.add_argument('-v', "--verbose", type=str, help='[y/N]')
parser.add_argument('-epoch_save', type=str, help='[y/N]')
parser.add_argument('-val', "--validate", type=str, help='[y/N]')
args = parser.parse_args()

# verbosity
verbose = False
if args.verbose == 'y' or args.verbose == 'Y':
    verbose = True

# device
device = torch.device('cpu')
if (args.cuda == 'y' or args.cuda == 'Y') and torch.cuda.is_available():
    device = torch.device('cuda')

if verbose:
    print(f'Device: {device}')


# model
resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
model = net.PetNet(resnet)

if verbose:
    print(f'Model Loaded!')

# Data Loaders
transform = Compose([
    ToTensor(),
    transforms.Resize((150,150))
])

# training
train_dataset = PetDataset(img_dir='images', training=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=int(args.batch_size), shuffle=False)

# validation
val_dataset = PetDataset(img_dir='images', training=True, transform=transform)
val_loader = DataLoader(dataset=val_dataset, batch_size=int(args.batch_size), shuffle=False)


# optimizer
learning_rate = 1e-4
weight_decay = 1e-5
model_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# scheduler
step_size = 7
gamma = 0.1
model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, step_size=step_size, gamma=gamma)

# loss function
loss_function = nn.MSELoss()

def train():

    # training flags for epoch saving/validation
    save_every_epoch = False
    if (args.epoch_save == 'y' or args.epoch_save) == 'Y':
        save_every_epoch = True
    validate = False
    if (args.epoch_save == 'y' or args.epoch_save) == 'Y':
        validate = True

    # set to training mode, send model to device
    model.to(device)
    epoch_losses_train = []
    epoch_losses_val = []

    # calculate number of batches
    n_batches_train = len(train_loader)/args.batch_size
    n_batches_val = len(val_loader)/args.batch_size

    for epoch in range(1, args.epochs+1):
        print(f'Epoch #{epoch}, Start Time: {datetime.datetime.now()}')
        loss_train = 0

        # training
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.squeeze().to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)

            # calculate losses
            loss = loss_function(outputs, labels)
            loss_train += loss.item()

            # backpropagation
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()

        epoch_losses_train += [loss_train / n_batches_train]
        print(f'Training Epoch {epoch} Loss: {epoch_losses_train[epoch - 1]}')
        model_scheduler.step()

        # validating
        if validate:
            loss_val = 0
            model.eval()
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.squeeze().to(device=device)
                    labels = labels.to(device=device)
                    labels = labels.long()
                    outputs = model(imgs)

                    # calculate losses
                    loss = loss_function(outputs, labels)
                    loss_val += loss.item()

            epoch_losses_val += [loss_val/n_batches_val]
            print(f'Validation Epoch {epoch} Loss: {epoch_losses_val[epoch - 1]}')

        if save_every_epoch:  # saving temporary model files and loss plots

            # FC Layers
            # create root directory if not there already
            model_folder_dir = './temp_models'
            if not os.path.isdir(model_folder_dir):
                os.mkdir(model_folder_dir)

            # save temp model
            try:
                temp_model_path = model_folder_dir + '/' + args.fc_output[:-4] + '_epoch' + str(epoch) + '.pth'
                torch.save(model.fc_layers.state_dict(), temp_model_path)
                if verbose:
                    print(f'Saved model for epoch {epoch} @{temp_model_path}')
            except:
                print('Epoch model save failed')

            # LOSS PLOT

            # create root directory if not there already
            plot_folder_dir = './temp_plots'
            if not os.path.isdir(plot_folder_dir):
                os.mkdir(plot_folder_dir)

            # save temp plot
            try:
                temp_plot_path = plot_folder_dir + '/' + args.loss_plot[:-4] + '_epoch' + str(epoch) + '.png'
                if validate:
                    generate_loss_plot_with_val(epoch_losses_train, epoch_losses_val, temp_plot_path)
                else:
                    generate_loss_plot(epoch_losses_train, temp_plot_path)
                if verbose:
                    print(f'Saved loss plot for epoch {epoch} @{temp_plot_path}')
            except:
                print('Epoch plot save failed')

        if verbose:
            print('\n')

    # save final frontend model
    torch.save(model.fc_layers.state_dict(), args.fc_output)

    if validate:
        generate_loss_plot_with_val(epoch_losses_train, epoch_losses_val, args.loss_plot, show_plot=True)
    else:
        generate_loss_plot(epoch_losses_train, args.loss_plot, show_plot=True)

    print('\n#########################################')
    print(f'Done training,'
          f'find final model file @{args.fc_output} '
          f'and final loss plot file @{args.loss_plot}')
    print('#########################################\n')


def generate_loss_plot(loss, file_loc, show_plot=False): # loss plot without validation
    epochs = list(range(1, len(loss)+1))
    plt.plot(epochs, loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.savefig(file_loc)
    if show_plot:
        plt.show()
    plt.close()


def generate_loss_plot_with_val(train_loss, val_loss, file_loc, show_plot=False): # loss plot with validation
    epochs = list(range(1, len(train_loss)+1))
    plt.plot(epochs, train_loss, label = "Training Loss")
    plt.plot(epochs, val_loss, label= "Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()
    plt.savefig(file_loc)
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    train()