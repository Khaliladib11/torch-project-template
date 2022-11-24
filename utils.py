import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



def get_loader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=None,
):
    """
    method to load dataloader
    :param dataset: the dataset object
    :param batch_size: batch size
    :param shuffle: Bool shuffle
    :param num_workers: number of worker (when using GPU)
    :return:
    """
    return DataLoader(dataset, batch_size, shuffle, num_workers)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    method to train the model one epoch
    :param model: the model itself
    :param loader: the dataloader
    :param criterion: the loss function
    :param optimizer: the optimizer
    :param device: cuda or cpu
    :return: the loss
    """
    model.train()
    training_loss = 0
    loop = tqdm(loader)
    for idx, batch in enumerate(loop):
        # image1, image2, degree = batch
        # image1, image2, degree = image1.to(device), image2.to(device), degree.to(device)
        image, degree = batch
        image, degree = image.to(device), degree.to(device)

        optimizer.zero_grad()
        # output = model(image1, image2)
        output = model(image)
        losses = criterion(output, degree)
        training_loss += losses.item()
        loop.set_postfix(loss=training_loss / (idx + 1))
        losses.backward()
        optimizer.step()

    training_loss /= len(loader)
    return training_loss


def validated_one_epoch(model, loader, criterion, device):
    """
    method to evaluate the model one epoch
    :param model: the model itself
    :param loader: the dataloader
    :param criterion: the loss function
    :param device: cuda or cpu
    :return: the loss
    """
    model.eval()
    val_loss = 0
    loop = tqdm(loader)
    for idx, batch in enumerate(loop):
        # image1, image2, degree = batch
        # image1, image2, degree = image1.to(device), image2.to(device), degree.to(device)
        # output = model(image1, image2)
        image, degree = batch
        image, degree = image.to(device), degree.to(device)
        output = model(image)
        losses = criterion(output, degree)
        val_loss += losses.item()
        loop.set_postfix(loss=val_loss / (idx + 1))

    val_loss /= len(loader)
    return val_loss


def logger(epoch, train_loss, val_loss, path_to_file, verbose=True):
    log_string = f"Epoch: {epoch}, Training Loss: {train_loss}, Val Loss: {val_loss}"

    if verbose:
        print(log_string)

    with open(path_to_file, 'a') as file:
        file.write(log_string + '\n')


def train(train_loader,
          val_loader,
          model,
          optimizer,
          criterion,
          device,
          epochs,
          path_to_save,
          checkpoint=None):
    print("######################################## Start Training ########################################")
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)
    logger_path = os.path.join(path_to_save, 'results.txt')
    best_path = os.path.join(path_to_save, 'best.pt')
    last_path = os.path.join(path_to_save, 'last.pt')

    if checkpoint:
        checkpoint_params = {
            'model': model,
            'optimizer': optimizer,
            'path': checkpoint
        }

        last_epoch, model, optimizer, train_loss, val_loss, best_val_loss = load_checkpoint(**checkpoint_params)

    else:
        train_loss = []
        val_loss = []
        best_val_loss = sys.maxsize
        last_epoch = 0

    model.to(device)
    for epoch in range(last_epoch, epochs):
        epoch_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_val_loss = validated_one_epoch(model, val_loader, criterion, device)

        train_loss.append(epoch_train_loss)
        val_loss.append(epoch_val_loss)
        logger(epoch, round(epoch_train_loss, 4), round(epoch_val_loss, 4), logger_path)
        if epoch_val_loss < best_val_loss:
            print("######################################## Model Improved ########################################")
            save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, best_path)
            best_val_loss = epoch_val_loss

        save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, last_path)

    return train_loss, val_loss


def evaluate(model, loader, criterion, device):
    test_loss = 0
    model.to(device)
    loop = tqdm(loader)
    for batch in loop:
        image1, image2, degree = batch
        image1, image2, degree = image1.to(device), image2.to(device), degree.to(device)
        output = model(image1, image2)
        losses = criterion(output, degree)
        loop.set_postfix(loss=losses.item())
        test_loss += losses.item()

    test_loss /= len(loader)
    return test_loss



def transform_(input):
    t = transforms.Compose([
        # transforms.Resize(image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std)
    ])
    return t(input)

@torch.no_grad()
def predict(input, model, device):
    """
    method to predict from an input
    :param input: the input, it could be string, image, etc...
    :param model: the model we want to use for prediction
    :param device: the device used for prediction, e.g.: cpu, cuda
    :return: prediction 
    """
    model.to(device)  # move model to device
    model.eval()  # put model in eval mode
    input = transform_(input)  # apply tranfrom
    input = input.unsqueeze(0)  # add batch size
    input = input.to(device)  # move input to device
    output = model(input)  # predict
    pred = output.argmax(dim=1)
    return pred.item()  # return output


def display_losses(train_loss, val_loss, xlabel="Iterations", ylabel="Loss", save_path=None):
    """
    method to display the training and validation losses
    :param train_loss: training loss
    :param val_loss: validation loss
    :param xlabel: xlabel string
    :param ylabel: ylabel string
    :return: None
    """
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    if save_path:
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, best_val_loss, path, verbose=False):
    torch.save({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

    if verbose:
        print("######################################## Model saved ########################################")


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    best_val_loss = checkpoint['best_val_loss']
    return epoch, model, optimizer, train_loss, val_loss, best_val_loss