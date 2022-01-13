# All the defenses I have implemented, adapted for the MNIST dataset
import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

# For interpolated adversarial training
import defenses.iat as iat

# For Jacobian Regularization
from jacobian import JacobianReg

# This here actually adds the path
sys.path.append("../")
import models.lenet as lenet


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# This method creates a new model and also trains it
def standard_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_standard"
):
    # Helps speed up operations
    scaler = torch.cuda.amp.GradScaler()

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

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Training Progress"):
            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Clean the gradients
                optimizer.zero_grad()

                # Predict
                logits = model(images)

                # Calculate loss
                with torch.cuda.amp.autocast():
                    loss = loss_function(logits, labels)

                # Gradient descent
                scaler.scale(loss).backward()

                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


# Adversarial examples should be typically generated when model parameters are not
# changing i.e. model parameters are frozen. This step may not be required for very
# simple linear models, but is a must for models using components such as dropout
# or batch normalization.
def adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_adversarial",
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


def interpolated_adversarial_training(
  trainSetLoader,
  attack_name,
  attack_function,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_interpolated_adversarial",
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


def dual_adversarial_training(
  trainSetLoader,
  attack_function1,
  attack_function2,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_dual",
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
                optimizer.step()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model


def jacobian_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_jacobian",
  **kwargs):
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
  load_path="../models_data/MNIST/mnist_interpolated_adversarial",
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