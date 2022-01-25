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
import models.lenet as lenet

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


def jacobian_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_jacobian",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = lenet.LeNet5().to(device)
    model.train()

    jacobian_reg = JacobianReg()
    jacobian_reg_lambda = 0.01

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
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
                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_alp",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

    # ALP factor
    alp_loss_function = nn.MSELoss()
    alp_lamda = 0.2

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = lenet.LeNet5().to(device)
    model.train()

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
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

                optimizer.step()

    print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def jacobian_ALP_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_alp",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

    # Jacobian Factor
    jacobian_reg = JacobianReg()
    jacobian_reg_lambda = 0.01

    # ALP factor
    alp_loss_function = nn.MSELoss()
    alp_lamda = 0.2

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = lenet.LeNet5().to(device)
    model.train()

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
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

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model
