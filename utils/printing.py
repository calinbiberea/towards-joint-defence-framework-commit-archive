import torch
import torch.nn as nn
import torchvision.utils

import numpy as np
import matplotlib.pyplot as plt


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


def print_image(image, title, plot):
    numpy_image = image.numpy()
    plot.imshow(np.transpose(numpy_image, (1, 2, 0)))
    plot.set_title(title)


def print_attack(model, testSetLoader, attack_name, attack_function, number_of_images=1, **kwargs):
    # Network parameters
    loss_function = nn.CrossEntropyLoss()

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

    # This is becase for each image, we want to also print the perturbed image
    number_columns = 2

    # Subplot(r,c) provide the number of rows and columns
    figure, axarr = plt.subplots(
        number_of_images,
        number_columns,
        figsize=(2 * number_columns, 2.5 * number_of_images),
    )
    figure.subplots_adjust(right=1)
    figure.subplots_adjust(hspace=1)

    # Check if using a library attack
    if "library" in kwargs:
        from_library = kwargs["library"]
    else:
        from_library = False

    if epsilon is not None:
        figure.suptitle("{} Attack using epsilon = {}".format(attack_name, epsilon))
    else:
        figure.suptitle("{} Attack".format(attack_name))

    # Get iterations
    if "iterations" in kwargs:
        iterations = kwargs["iterations"]
    else:
        iterations = None

    # Select the images and show the noise
    correct_image_broken = 0
    while True:
        # Get random image index
        index = np.random.randint(0, len(testSetLoader.dataset))

        # Get an image and cast it to CUDA if needed, cast to proper batches
        image, label = testSetLoader.dataset[index]
        image = image[None, :]
        label = torch.as_tensor((label,))

        image, label = image.to(device), label.to(device)

        # Predict
        logits = model(image)
        _, pred = torch.max(logits, 1)

        # Only count correct images
        if pred != label:
            continue

        # Perturb the images using the attack
        if not from_library:
            perturbed_image = attack_function(
                image,
                label,
                model,
                loss_function,
                epsilon=epsilon,
                alpha=alpha,
                scale=True,
                iterations=iterations,
            )
        else:
            perturbed_image = attack_function(image, label)

        # Calculate results
        logits = model(perturbed_image)
        _, fgsm_pred = torch.max(logits, 1)

        pred = pred.cpu().detach()[0]
        fgsm_pred = fgsm_pred.cpu().detach()[0]

        # Get the plots
        if number_of_images == 1:
            image_plot = axarr[0]
            perturbed_image_plot = axarr[1]
        else:
            image_plot = axarr[correct_image_broken, 0]
            perturbed_image_plot = axarr[correct_image_broken, 1]

        # Print the original image
        print_image(
            torchvision.utils.make_grid(image.cpu().data, normalize=True),
            f"Predicted {testSetLoader.dataset.classes[pred]}",
            image_plot,
        )

        # Print the perturbed iamge
        print_image(
            torchvision.utils.make_grid(perturbed_image.cpu().data, normalize=True),
            f"Predicted {testSetLoader.dataset.classes[fgsm_pred]}",
            perturbed_image_plot,
        )

        # Only count correctly predicted images that got tricked
        correct_image_broken += 1
        if correct_image_broken >= number_of_images:
            break


# Helper function for plotting the gradients
def print_gradient(gradient):
    plt.style.use("classic")
    numpy_gradient_sign = np.sign(gradient.cpu().detach().numpy())
    plt.imshow(np.transpose(numpy_gradient_sign, (1, 2, 0)))
    plt.show()