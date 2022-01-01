# If I really want to use a library, there is this
# https://advertorch.readthedocs.io/en/latest/advertorch/attacks.html#advertorch.attacks.GradientSignAttack

import torch

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


# FGSM attack code
def fgsm_attack(
  images,
  labels,
  model,
  loss_function,
  epsilon,
  scale=False,
  **kwargs):
    # Clamp value (i.e. make sure pixels lie in 0-255)
    clamp_max = 255

    # Adding clipping to maintain [0,1] range if that is the scale
    if scale:
        clamp_max = clamp_max / 255

    # Make sure gradient is actually compute
    images.requires_grad = True
    logits = model(images)

    # Ensure model stays unchanged, then calculate loss and gradients
    model.zero_grad()
    loss = loss_function(logits, labels).to(device)
    loss.backward()

    # Create the perturbed image by adjusting each pixel of the input images
    perturbed_image = images + epsilon * images.grad.sign()

    # Make sure pixels' values lie in correct range
    perturbed_image = torch.clamp(perturbed_image, min=0, max=clamp_max)

    # Return the perturbed images
    return perturbed_image