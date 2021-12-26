# If I really want to use a library, there is this
# https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.GradientSignAttack

import torch


# FGSM attack code
def fgsm_attack(images, epsilon, data_grad):

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * sign_data_grad

    # Adding clipping to maintain [0,1] range
    perturbed_images = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image
    return perturbed_images