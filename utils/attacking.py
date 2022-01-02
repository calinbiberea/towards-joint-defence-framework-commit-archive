import torch
import torch.nn as nn
from tqdm.notebook import tqdm


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


def attack_model(model, testSetLoader, attack_name, attack_function, **kwargs):
    # Network parameters
    loss_function = nn.CrossEntropyLoss()

    correct = 0

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

    if epsilon is not None:
        print(
            "Testing the model under {} Attack using epsilon = {}, alpha = {}...".format(
                attack_name, epsilon, alpha
            )
        )
    else:
        print("Testing the model under {} Attack...".format(attack_name))

    # Get iterations
    if "iterations" in kwargs:
        iterations = kwargs["iterations"]
    else:
        iterations = None

    # Check if using a library attack
    if "library" in kwargs:
        from_library = kwargs["library"]
    else:
        from_library = False

    # Use a pretty progress bar to show updates
    for j, (images, labels) in enumerate(
        tqdm(testSetLoader, desc="{} Attack Testing Progress".format(attack_name))
    ):
        # Cast to proper tensor
        images, labels = images.to(device), labels.to(device)

        # Perturb the images using the attack
        if not from_library:
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
        else:
            perturbed_images = attack_function(images, labels)

        # Calculate results
        logits = model(perturbed_images)

        _, preds = torch.max(logits, 1)

        correct += (preds == labels).sum().item()

    print("... done! Accuracy: {}%\n\n".format(float(correct) * 100 / 10000))