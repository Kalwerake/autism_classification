import torch
import numpy as np


def train_one_epoch_binary(model, loss_fn, optimiser, train_loader, device):
    """
    Function for training one epoch
    model: model class (pytorch module)
    loss_fn: loss function, (pytorch module)
    optimiser: optimser (pytorch module)
    train_loader: train set dataloader (pytorch DataLoader class)
    """
    running_loss = 0
    epoch_accuracy = []
    for j, (x_train, y) in enumerate(train_loader):
        optimiser.zero_grad()  # zero the gradient at each epoch start
        y = y.to(device)  # send y to cuda
        x_train = x_train.to(device)
        prediction = model.forward(x_train)
        loss = loss_fn(prediction, y)  # loss

        accuracy = (torch.round(
            prediction) == y).float().mean()  # calculate accuracy for each mini-batch  take prediction tensor, reshape to 1d detach from computational graph turn to numpy array, round and see if rounded number is equal to label, find mean of this boolean array, this is the accuracy

        running_loss += loss.item()  # get epoch loss
        epoch_accuracy.append(accuracy.item())

        loss.backward()  # backward propgation
        optimiser.step()

        running_loss += loss.item()  # get epoch loss
        epoch_accuracy.append(accuracy.item())

    return running_loss, np.mean(epoch_accuracy)


def validate_one_epoch_binary(model, loss_fn, test_loader, device):
    """
       Function for training one epoch
       model: model class (pytorch module)
       loss_fn: loss function, (pytorch module)
       train_loader: test set dataloader (pytorch DataLoader class)
       """
    test_loss_run = 0
    test_acc_epoch = []
    for j, (x_test, y_test) in enumerate(test_loader):
        y_test = y_test.to(device)
        x_test = x_test.to(device)
        test_pred = model.forward(x_test)
        test_loss = loss_fn(test_pred, y_test)  # loss

        test_acc = (torch.round(
            test_pred) == y_test).float().mean()  # calculate accuracy for each mini-batch  take prediction tensor, reshape to 1d detach from computational graph turn to numpy array, round and see if rounded number is equal to label, find mean of this boolean array, this is the accuracy

        test_loss_run += test_loss.item()
        test_acc_epoch.append(test_acc.item())

    return test_loss_run, np.mean(test_acc_epoch)


class EarlyStopper:
    """
    EarlyStopper module
    """
    def __init__(self, patience=1, min_delta=0):
        """
        patience: (int) number of epochs to tolerate increase in validation loss
        min_delta: (int or flt) threshold for validation loss
        """

        self.patience = patience
        self.min_delta = min_delta
        # number of epochs where validation loss increases, start at 0
        self.counter = 0
        # iniatlise miniumum validation loss as positive infinity
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        # if validation loss is is lower than current min validation loss
        if validation_loss < self.min_validation_loss:
            # set min validation loss to current validation loss
            self.min_validation_loss = validation_loss
            # dont add to counter
            self.counter = 0
        # else if validation loss is greater than the current min validation loss plus the tolerence
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # add one to counter, indicate that this epoch the validation loss has not decreased
            self.counter += 1
            # if the counter reaches the patience threshold return True, this stops traininf
            if self.counter >= self.patience:
                return True
        # if above elif condition is not met return false, training continues
        return False
