import torch

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


# I-FGSM attack code
def ifgsm_attack(
  images,
  labels,
  model,
  loss_function,
  epsilon,
  alpha,
  iterations=0,
  scale=False
):
    # Clamp value (i.e. make sure pixels lie in 0-255)
    clamp_max = 255

    # Adding clipping to maintain [0,1] range if that is the scale
    if scale:
        clamp_max = clamp_max / 255

    # The paper gives this formula for managing iterations
    if iterations == 0:
        iterations = int(min(epsilon + 4, 1.25 * epsilon))

    for iteration in range(iterations):
        images.requires_grad = True
        outputs = model(images)

        # Ensure model stays unchanged, then calculate loss and gradients
        model.zero_grad()
        loss = loss_function(outputs, labels).to(device)
        loss.backward()

        fgsm_images = images + alpha * images.grad.sign()

        # Clipping part of the basic iterative method
        # Use 'eps' instead of epsilon to save space
        # Clip attack using X'' = min{255, X + eps, max{0, X - eps, X'}}
        # which is the same as = min{255, min{X + eps, max{max{0, X - eps}, X'}}}

        # max{0, X - eps}
        step1 = torch.clamp(images - epsilon, min=0)

        # max{max{0, X - eps}, X'}
        step2 = (fgsm_images >= step1).float() * fgsm_images + (step1 > fgsm_images).float() * step1

        # min{X + eps, max{max{0, X - eps}, X'}}
        step3 = (step2 > (images + epsilon)).float() * (images + epsilon) + (images + epsilon >= step2).float() * step2

        # min{255, min{X + eps, max{max{0, X - eps}, X'}}}
        images = torch.clamp(step3, max=clamp_max).detach_()
        images = images.to(device)

    # Return the perturbed images
    return images