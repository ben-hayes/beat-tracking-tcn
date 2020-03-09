"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/utils/training.py

Description: Provides utilities for training a BeatNet model on a given
             dataset.
"""


def train(
        model,
        criterion,
        optimiser,
        data_loader,
        cuda_device=None,
        batch_callback=None):
    """
    Trains a BeatNet model for one epoch using the given data, loss function,
    and optimiser.

    Arguments:
        model {torch.nn.Module} -- The PyTorch model to be trained.
        criterion {torch.nn._Loss} -- The loss function to use for training.
        optimiser {torch.optim.Optimizer} -- The optimiser to be used for
                                             training.
        data_loader {torch.utils.data.DataLoader} -- The PyTorch DataLoader
                                                     representing the training
                                                     data set.

    Keyword Arguments:
        cuda_device {torch.device} -- The CUDA device to use for training.
                                      Training defaults to CPU if none is
                                      specified. (default: {None})
        batch_callback {function} -- A function to be called after each batch
                                     to allow for logging or printing of loss.
                                     (default: {None})

    Returns:
        dict -- A dict containing fields "total_batches" and "epoch_loss".
    """
    model.train()

    running_loss = 0.0

    for i, batch in enumerate(data_loader):
        optimiser.zero_grad()

        outputs, loss = forward_batch(model, criterion, batch, cuda_device)

        loss.backward()
        optimiser.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        if batch_callback is not None:
            batch_callback({
                "batch_loss": batch_loss,
                "batch_index": i,
                "running_epoch_loss": running_loss / (i + 1)
            })

    return {
        "total_batches": i + 1,
        "epoch_loss": running_loss / (i + 1)
    }


def evaluate(
        model,
        criterion,
        data_loader,
        cuda_device=None,
        batch_callback=None,
        evaluation_functions=None):
    """
    Evaluate the performance of a BeatNet function according to a given loss
    function and, optionally, a list of evaluation callbacks.

    Arguments:
        model {torch.nn.Module} -- The PyTorch model to be evaluated.
        criterion {torch.nn._Loss} -- The loss function to use for evaluation.
        data_loader {torch.utils.data.DataLoader} -- The PyTorch DataLoader
                                                     representing the desired
                                                     data set.

    Keyword Arguments:
        cuda_device {torch.device} -- The CUDA device to use for evaluation.
                                      Evaluation defaults to CPU if none is
                                      specified. (default: {None})
        batch_callback {function} -- A function to be called after each batch
                                     to allow for logging or printing of loss.
                                     (default: {None})
        evaluation_functions {list(function)} -- A list of functions to call
                                                 for evaluation. Functions
                                                 should take arguments
                                                 (prediction, target).
                                                 (default: {None})

    Returns:
        dict -- A dict containing "total_batches", "epoch_loss" and
                "running_evaluations".
    """
    model.eval()

    running_loss = 0.0

    if evaluation_functions is not None:
        running_evaluations = {
            func.__name__: 0.0
            for func in evaluation_functions
        }

    for i, batch in enumerate(data_loader):
        outputs, loss = forward_batch(model, criterion, batch, cuda_device)

        batch_loss = loss.item()
        running_loss += batch_loss

        if evaluation_functions is not None:
            evaluations = {
                func.__name__: func(outputs, batch["target"])
                for func in evaluation_functions
            }

            for func in evaluation_functions:
                running_evaluations[func.__name__] +=\
                    evaluations[func.__name__]

        if batch_callback is not None:
            batch_callback({
                "batch_loss": batch_loss,
                "batch_index": i,
                "running_epoch_loss": running_loss / (i + 1),
                "evaluations":
                    evaluations if evaluation_functions is not None else None
            })

    running_evaluations = {
        func: running_evaluations[func] / (i + 1)
        for func in running_evaluations
    } if evaluation_functions is not None else None

    return {
        "total_batches": i + 1,
        "epoch_loss": running_loss / (i + 1),
        "running_evaluations": running_evaluations 
    }


def forward_batch(
        model,
        criterion,
        batch,
        cuda_device=None):
    """
    Pass a batch through a PyTorch model, and calculate the loss.

    Arguments:
        model {torch.nn.Module} -- The model to feed data through.
        criterion {torch.optim.Optimizer} -- The loss function to use.
        batch {dict} -- The data batch.

    Keyword Arguments:
        cuda_device {torch.device} -- The CUDA device for training. CPU is used
                                      if none is specified. (default: {None})

    Returns:
        tuple -- A tuple of the output tensor and the calculated loss.
    """
    X = batch["spectrogram"]
    y = batch["target"]

    if cuda_device is not None:
        X = X.to(device=cuda_device)
        y = y.to(device=cuda_device)

    outputs = model(X)
    loss = criterion(outputs, y)

    return outputs, loss
