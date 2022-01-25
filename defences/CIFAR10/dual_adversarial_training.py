# Unlike the other datasets, CIFAR-10 uses ResNet and suffers from
# a variety of problems, including exploding gradients
import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

import torchattacks

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


def dual_adversarial_training(
  trainSetLoader,
  attack_function1,
  attack_function2,
  long_training=True,
  load_if_available=False,
  load_path="../models_data/CIFAR10/cifar10_dual",
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
        if "epsilon1" in kwargs:
            epsilon1 = kwargs["epsilon1"]
        else:
            epsilon1 = None

        # Check if using epsilon
        if "epsilon1" in kwargs:
            epsilon2 = kwargs["epsilon1"]
        else:
            epsilon2 = None

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

        use_cw = False

        # Sanity check to use CW
        if attack_function2 is None:
            use_cw = True

            if "steps" in kwargs:
                steps = kwargs["steps"]
            else:
                steps = 1000

            # Check if more epochs suplied
            if "c" in kwargs:
                c = kwargs["c"]
            else:
                c = 1000

            # Define the attack
            attack_function2 = torchattacks.CW(model, c=c, steps=steps)

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
            # Adjust the learning rate
            adjust_learning_rate(optimizer, epoch, learning_rate, long_training)

            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Run the attack
                model.eval()
                perturbed_images1 = attack_function1(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon1,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )

                # Include CW checks to allow smarter uses
                if not use_cw:
                    perturbed_images2 = attack_function2(
                        images,
                        labels,
                        model,
                        loss_function,
                        epsilon=epsilon2,
                        alpha=alpha,
                        scale=True,
                        iterations=iterations,
                    )
                else:
                    perturbed_images2 = attack_function2(
                        images,
                        labels,
                    )
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits1 = model(perturbed_images1)
                logits2 = model(perturbed_images2)
                loss1 = loss_function(logits1, labels)
                loss2 = loss_function(logits2, labels)

                loss = (loss1 + loss2) / 2

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


def triple_adversarial_training(
  trainSetLoader,
  attack_function1,
  attack_function2,
  attack_function3,
  long_training=True,
  load_if_available=False,
  load_path="../models_data/CIFAR10/cifar10_triple",
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
                perturbed_images1 = attack_function1(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )
                perturbed_images2 = attack_function2(
                    images,
                    labels,
                    model,
                    loss_function,
                    epsilon=epsilon,
                    alpha=alpha,
                    scale=True,
                    iterations=iterations,
                )
                # The third attack is imported from library (DeepFool or C&W)
                perturbed_images3 = attack_function3(
                    images,
                    labels,
                )
                model.train()

                # Predict and optimise
                optimizer.zero_grad()

                logits1 = model(perturbed_images1)
                logits2 = model(perturbed_images2)
                logits3 = model(perturbed_images3)
                loss1 = loss_function(logits1, labels)
                loss2 = loss_function(logits2, labels)
                loss3 = loss_function(logits3, labels)

                loss = (loss1 + loss2 + loss3) / 3

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