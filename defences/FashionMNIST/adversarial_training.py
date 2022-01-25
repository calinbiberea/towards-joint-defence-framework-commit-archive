import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

import torchattacks

# This here actually adds the path
sys.path.append("../../")
import defences.utils.iat as iat
import models.lenet as lenet

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# Adversarial examples should be typically generated when model parameters are not
# changing i.e. model parameters are frozen. This step may not be required for very
# simple linear models, but is a must for models using components such as dropout
# or batch normalization.
def adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/FashionMNIST/fashion_mnist_adversarial",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

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

        # Check if more epochs suplied
        if "epochs" in kwargs:
            epochs = kwargs["epochs"]

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

                logits = model(perturbed_images)
                loss = loss_function(logits, labels)

                # Gradient descent
                loss.backward()

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def cw_adversarial_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../models_data/FashionMNIST/fashion_mnist_cw",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

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

        # Check if more epochs suplied
        if "steps" in kwargs:
            steps = kwargs["steps"]
        else:
            steps = 1000

        # Check if more epochs suplied
        if "c" in kwargs:
            c = kwargs["c"]
        else:
            c = 1000

        # Check if more epochs suplied
        if "epochs" in kwargs:
            epochs = kwargs["epochs"]

        # Define the attack
        attack_function = torchattacks.CW(model, c=c, steps=steps)

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Run the attack
                model.eval()
                perturbed_images = attack_function(images, labels)
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits = model(perturbed_images)
                loss = loss_function(logits, labels)

                # Gradient descent
                loss.backward()

                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def interpolated_adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/FashionMNIST/fashion_mnist_interpolated_adversarial",
  **kwargs
):
    # Various training parameters
    epochs = 20
    learning_rate = 0.01

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

                # Make sure previous step gradients are not used
                optimizer.zero_grad()

                # Use manifold mixup to modify the data
                (
                    benign_mix_images,
                    benign_mix_labels_a,
                    benign_mix_labels_b,
                    benign_mix_lamda,
                ) = iat.mix_inputs(1, images, labels)

                # Predict and calculate benign loss
                benign_logits = model(benign_mix_images)
                benign_loss = iat.mixup_loss_function(
                    loss_function,
                    benign_mix_lamda,
                    benign_logits,
                    benign_mix_labels_a,
                    benign_mix_labels_b,
                )

                # Run the adversarial attack
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

                # Use manifold mixup on the adversarial data
                (
                    adversarial_mix_images,
                    adversarial_mix_labels_a,
                    adversarial_mix_labels_b,
                    adversarial_mix_lamda,
                ) = iat.mix_inputs(1, perturbed_images, labels)

                # Predict and calculate adversarial loss
                adversarial_logits = model(adversarial_mix_images)
                adversarial_loss = iat.mixup_loss_function(
                    loss_function,
                    adversarial_mix_lamda,
                    adversarial_logits,
                    adversarial_mix_labels_a,
                    adversarial_mix_labels_b,
                )

                # Take average of the two losses
                loss = (benign_loss + adversarial_loss) / 2

                # Gradient descent
                loss.backward()
                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model
