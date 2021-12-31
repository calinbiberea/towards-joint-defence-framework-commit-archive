import torch

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


def pgd_attack(
  images,
  labels,
  model,
  loss_function,
  epsilon=0.3,
  alpha=(2 / 255),
  iterations=20,
  scale=False,
  **kwargs
):
    # Clamp value (i.e. make sure pixels lie in 0-255)
    clamp_max = 255

    # Adding clipping to maintain [0,1] range if that is the scale
    if scale:
        clamp_max = clamp_max / 255

    images = images.to(device)
    labels = labels.to(device)

    original_images = images.data

    # PGD suggests random start (so it differs from BIM)
    # Starting at a uniformly random point in the given epsilon range
    images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    images = torch.clamp(images, max=clamp_max).detach()

    for iteration in range(iterations):
        images.requires_grad = True
        logits = model(images)

        # Ensure model stays unchanged, then calculate loss and gradients
        model.zero_grad()
        loss = loss_function(logits, labels).to(device)
        loss.backward()

        # Construct the adversarial images (in the iteration step)
        fgsm_images = images + alpha*images.grad.sign()

        # This is basically clamping since this is what works for l-infinity
        eta = torch.clamp(fgsm_images - original_images, min=-epsilon, max=epsilon)

        # Clamp again to keep in the right pixel (possibly normalised) intervals
        images = torch.clamp(original_images + eta, max=clamp_max).detach_()

    return images