# If I really want to use a library, there is this
# https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.GradientSignAttack

import torch


# FGSM attack code
def fgsm_attack(images, epsilon, data_grad, scale=False):
    # Clamp value (i.e. make sure pixels lie in 0-255)
    clamp_max = 255

    # Adding clipping to maintain [0,1] range if that is the scale
    if scale:
        clamp_max = clamp_max / 255

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input images
    perturbed_image = images + epsilon * sign_data_grad

    # Make sure pixels' values lie in correct range
    perturbed_image = torch.clamp(perturbed_image, max=clamp_max)

    # Return the perturbed images
    return perturbed_image