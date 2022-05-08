# Unlike the other datasets, CIFAR-10 uses ResNet and suffers from
# a variety of problems, including exploding gradients
import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

# For Jacobian Regularization
from jacobian import JacobianReg

# This here actually adds the path
sys.path.append("../../")
import models.resnet as resnet

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# Helps adjust learning rate for better results
def adjust_learning_rate(optimizer, epoch, learning_rate, long_training):
    actual_learning_rate = learning_rate
    if long_training:
        first_update_threshold = 100
        second_update_threshold = 150
    else:
        first_update_threshold = 20
        second_update_threshold = 25

    if epoch >= first_update_threshold:
        actual_learning_rate = 0.01
    if epoch >= second_update_threshold:
        actual_learning_rate = 0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = actual_learning_rate


def jacobian_training(
  trainSetLoader,
  long_training=True,
  load_if_available=False,
  load_path="../models_data/CIFAR10/cifar10_jacobian",
  **kwargs
):
    # Number of epochs is decided by training length
    if long_training:
        epochs = 200
    else:
        epochs = 30

    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()

    jacobian_reg = JacobianReg()
    if "jac_lambda" in kwargs:
        jacobian_reg_lambda = kwargs["jac_lambda"]
    else:
        jacobian_reg_lambda = 0.01

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Jacobian Regularization Training Progress"):
            # Adjust the learning rate
            adjust_learning_rate(optimizer, epoch, learning_rate, long_training)

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Require gradients for Jacobian regularization
                images.requires_grad = True

                # Predict and optimise
                optimizer.zero_grad()

                # Predict
                logits = model(images)

                # Calculate loss
                loss = loss_function(logits, labels)

                # Introduce Jacobian regularization
                jacobian_reg_loss = jacobian_reg(images, logits)

                # Total loss
                loss = loss + jacobian_reg_lambda * jacobian_reg_loss

                # Gradient descent
                loss.backward()

                # Also clip the gradients (ReLU leads to vanishing or
                # exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  long_training=True,
  load_if_available=False,
  load_path="../models_data/CIFAR10/cifar10_alp",
  **kwargs
):
    # Number of epochs is decided by training length
    if long_training:
        epochs = 200
    else:
        epochs = 30

    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()

    # ALP factor
    alp_loss_function = nn.MSELoss()
    alp_lamda = 0.2

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Check if using epsilon
        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = None

        # Check if using alpha
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = None

        # Get iterations
        if "iterations" in kwargs:
            iterations = kwargs["iterations"]
        else:
            iterations = None

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            # Adjust the learning rate
            adjust_learning_rate(optimizer, epoch, learning_rate, long_training)

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Run the attack
                model.eval()
                perturbed_images = attack_function(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits = model(images)
                loss = loss_function(logits, labels) + alp_lamda * alp_loss_function(
                    model(images), model(perturbed_images)
                )

                # Gradient descent
                loss.backward()

                # Also clip the gradients (ReLU leads to vanishing or
                # exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def jacobian_ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  long_training=True,
  load_if_available=False,
  load_path="../models_data/FashionMNIST/fashion_mnist_alp",
  **kwargs
):
    # Number of epochs is decided by training length
    if long_training:
        epochs = 200
    else:
        epochs = 30

    learning_rate = 0.1

    # Jacobian Factor
    jacobian_reg = JacobianReg()
    jacobian_reg_lambda = 0.01

    # ALP factor
    alp_loss_function = nn.MSELoss()
    alp_lamda = 0.2

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Check if using epsilon
        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = None

        # Check if using alpha
        if "alpha" in kwargs:
            alpha = kwargs["alpha"]
        else:
            alpha = None

        # Get iterations
        if "iterations" in kwargs:
            iterations = kwargs["iterations"]
        else:
            iterations = None

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            # Adjust the learning rate
            adjust_learning_rate(optimizer, epoch, learning_rate, long_training)

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Require gradients for Jacobian regularization
                images.requires_grad = True

                # Run the attack
                model.eval()
                perturbed_images = attack_function(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits = model(images)
                loss = loss_function(logits, labels) + alp_lamda * alp_loss_function(
                    model(images), model(perturbed_images)
                )

                # Introduce Jacobian regularization
                jacobian_reg_loss = jacobian_reg(images, logits)

                # Total loss
                loss = loss + jacobian_reg_lambda * jacobian_reg_loss

                # Gradient descent
                loss.backward()

                # Also clip the gradients (ReLU leads to vanishing or
                # exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model