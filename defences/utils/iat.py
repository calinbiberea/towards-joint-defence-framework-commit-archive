# Additional helpers for constructing an interpolated adversarial example
# Paper: https://arxiv.org/pdf/1906.06784.pdf

import numpy as np
import torch


# Mixes up a set of iamges and their labels
# In particular, it's an implemention of manifold mixup (https://arxiv.org/abs/1806.05236)
def mix_inputs(mixup_constant, inputs, labels):
    # Draw from a Beta distribution
    mix_lambda = np.random.beta(mixup_constant, mixup_constant)

    # Get the batch size (easier than passing it as an argument)
    batch_size = inputs.size()[0]

    # Make a random permutation
    index = torch.randperm(batch_size)

    # Mix the items up
    mixed_x = mix_lambda * inputs + (1 - mix_lambda) * inputs[index, :]

    # Split the labels
    labels_a, labels_b = labels, labels[index]

    # Return the result
    return mixed_x, labels_a, labels_b, mix_lambda


# Caculates the mixup loss given a particular loss function
def mixup_loss_function(loss_function, mix_lambda, logits, labels_a, labels_b):
    return mix_lambda * loss_function(logits, labels_a) + (1 - mix_lambda) * loss_function(logits, labels_b)