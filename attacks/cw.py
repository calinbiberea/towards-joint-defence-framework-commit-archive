import torch

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


# The helper functions that are being used in the paper
def tanh_space(x):
    return 1 / 2 * (torch.tanh(x) + 1)


def inverse_tanh_space(x):
    return torch.atanh(x * 2 - 1)


# The 'f' function that was given in the paper
def f(logits, labels, kappa, targeted=False):
    one_hot_labels = torch.eye(len(logits[0]))[labels].to(device)

    # We need the second largest logit
    next_largest_logit, _ = torch.max((1-one_hot_labels) * logits, dim=1)

    # We also need the largest logit
    largest_logit = torch.masked_select(logits, one_hot_labels.bool())

    # Decide what to do based on if the attack is targeted
    if targeted:
        return torch.clamp((next_largest_logit - largest_logit), min=-kappa)
    else:
        return torch.clamp((largest_logit - next_largest_logit), min=-kappa)


# Untargeted version of the attack
def cw_attack(
  images,
  labels,
  model,
  loss_function,
  iterations=50,
  **kwargs
):
    # Default values for the parameters given in the paper
    # While binary search was useful, I have not implemented
    # it since it did not bring in additional value
    lr = 0.01
    c = 1e-4
    kappa = 0

    images = images.to(device)
    labels = labels.to(device)

    # Get the 'w' from the paper (essentially a change of variable)
    w = inverse_tanh_space(images).detach()
    w.requires_grad = True

    # Use some additional variables to track the best results (so far)
    best_adversarial_images = images.clone().detach()
    best_L2_norm = 1e10 * torch.ones((len(images))).to(device)
    prev_total_loss = 1e10

    # Constant for size
    dim = len(images.shape)

    # The two components that we need for calculating loss
    MSELoss = torch.nn.MSELoss(reduction='none')
    Flatten = torch.nn.Flatten()

    # We need an optimizer on 'w' since we did a change of variable
    optimizer = torch.optim.Adam([w], lr=lr)

    for iteration in range(iterations):
        # Computer adversarial images
        adversarial_images = tanh_space(w)

        # Calculate loss
        current_L2_norm = MSELoss(Flatten(adversarial_images), Flatten(images)).sum(dim=1)
        L2_loss = current_L2_norm.sum()

        # Predict
        logits = model(adversarial_images)

        # Calculate the loss
        f_loss = f(logits, labels, kappa).sum()

        # Gradient descent to find next adversarial image update
        total_loss = L2_loss + c * f_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update adversarial images
        _, pred = torch.max(logits.detach(), 1)
        correct = (pred == labels).float()

        # Filter out images that get either correct predictions or non-decreasing loss,
        # keep only images that are both misclassified and loss-decreasing are left
        mask = (1 - correct) * (best_L2_norm > current_L2_norm.detach())
        best_L2_norm = mask * current_L2_norm.detach() + (1 - mask) * best_L2_norm

        # Update the view of the mask to update the adversarial images
        mask = mask.view([-1] + [1] * (dim - 1))
        best_adversarial_images = mask * adversarial_images.detach() + (1 - mask) * best_adversarial_images

        # If loss does not converge, there is no value in continu
        if iteration % (iterations // 10) == 0:
            if total_loss.item() > prev_total_loss:
                return best_adversarial_images

            prev_total_loss = total_loss.item()

    return best_adversarial_images
