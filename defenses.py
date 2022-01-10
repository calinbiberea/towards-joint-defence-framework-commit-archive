#!/usr/bin/env python
# coding: utf-8

# # CIFAR-10: Adversarial Training and Defenses

# ## Imports and CIFAR-10 loading

# In[18]:


# For plotting
import numpy as np
import torch
import torch.nn as nn

# Nice loading bars
from tqdm.notebook import tnrange, tqdm

# DNN used
import models.resnet as resnet

# Test the loaded model
import utils.clean_test as clean_test


# In[19]:


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# ## Clean Datasets

# In[20]:


# Get the data loaders (assume we do no validation)
import utils.dataloaders as dataloaders

DATA_ROOT = "./datasets/CIFAR10"

trainSetLoader, _, testSetLoader = dataloaders.get_CIFAR10_data_loaders(
    DATA_ROOT,
    trainSetSize=50000,
    validationSetSize=0,
    batchSize=128,
)


# ## Adversarial Training

# In[21]:


import attacks.fgsm as fgsm
import attacks.pgd as pgd

attacks = {}
attacks["FGSM"] = fgsm.fgsm_attack
attacks["PGD"] = pgd.pgd_attack

import utils.attacking as attacking

# For printing outcomes
import utils.printing as printing


# In[22]:


# Adversarial examples should be typically generated when model parameters are not
# changing i.e. model parameters are frozen. This step may not be required for very
# simple linear models, but is a must for models using components such as dropout
# or batch normalization.


# Note: to speed up training, using this https://arxiv.org/abs/2001.03994 variant
def get_adversarially_trained_model(attack, **kwargs):
    # For 200 epochs, use 150, 200 as the epochs tresholds
    # Also learning rates are 0.05, 0.01, 0.001

    # Helps adjust learning rate for better results
    def adjust_learning_rate(optimizer, epoch, learning_rate):
        actual_learning_rate = learning_rate
        if epoch >= 150:
            actual_learning_rate = 0.01
        if epoch >= 200:
            actual_learning_rate = 0.001
        for param_group in optimizer.param_groups:
            param_group["lr"] = actual_learning_rate

    # Various training parameters
    epochs = 200
    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

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

    # Get the attack
    attack_function = attacks[attack]

    print("Training the model using adversarial examples...")
    model.train()

    # Use a pretty progress bar to show updates
    for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        total_epoch_loss = 0

        for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
            # Cast to proper tensors
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Run the attack
            model.eval()
            perturbed_images = attack_function(
                images,
                labels,
                model,
                nn.CrossEntropyLoss(),
                epsilon=epsilon,
                alpha=alpha,
                scale=True,
                iterations=iterations,
            )
            model.train()

            # Predict and optimise
            logits = model(perturbed_images)
            loss = loss_function(logits, labels)

            # Early stopping if loss becomes nan
            if np.isnan(loss.item()):
                print("...terminating early due to loss being nan...")

                return model

            # Track total loss
            total_epoch_loss += loss.item()

            # Gradient descent
            loss.backward()

            # Also clip the gradients (for some reason they super explode and loss becomes nan)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        # To track if the model is getting better
        print("Loss is ", total_epoch_loss)
        model.eval()
        clean_test.test_trained_model(model, testSetLoader)
        attacking.attack_model(
            model,
            testSetLoader,
            "FGSM",
            attacks["FGSM"],
            epsilon=epsilon,
        )
        model.train()

    print("... done!")

    # Return the trained model
    return model


# In[23]:


# Quite similar to the one above, but this one introduces IAT (Interpolated Adversarial Training)
import defenses.iat as iat


def get_interpolated_adversarially_trained_model(attack, **kwargs):
    # Helps adjust learning rate for better results
    def adjust_learning_rate(optimizer, epoch, learning_rate):
        actual_learning_rate = learning_rate
        if epoch >= 150:
            actual_learning_rate = 0.01
        if epoch >= 200:
            actual_learning_rate = 0.001
        for param_group in optimizer.param_groups:
            param_group["lr"] = actual_learning_rate

    # Various training parameters
    epochs = 200
    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002
    )

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

    # Get the attack
    attack_function = attacks[attack]

    print("Training the model using adversarial examples...")
    model.train()

    # Use a pretty progress bar to show updates
    for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
        adjust_learning_rate(optimizer, epoch, learning_rate)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        # To track if the model is getting better
        model.eval()
        clean_test.test_trained_model(model, testSetLoader)
        attacking.attack_model(
            model,
            testSetLoader,
            "FGSM",
            attacks["FGSM"],
            epsilon=epsilon,
        )
        model.train()

    print("... done!")

    # Return the trained model
    return model


# ### FGSM Adversarial Training

# In[24]:


fgsm_model = get_adversarially_trained_model("FGSM", epsilon=0.075)

# From this point on, we simply want to evaluate
fgsm_model.eval()
clean_test.test_trained_model(fgsm_model, testSetLoader)


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 0.1, 0.2, 0.35, 0.55]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        fgsm_model,
        testSetLoader,
        "FGSM",
        attacks["FGSM"],
        epsilon=epsilon,
    )


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 4 / 255, 0.1, 0.3]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        fgsm_model,
        testSetLoader,
        "PGD",
        attacks["PGD"],
        epsilon=epsilon,
        alpha=(2 / 255),
        iterations=20,
    )


# In[ ]:


# Make sure to test the final accuracy of the model
clean_test.test_trained_model(fgsm_model, testSetLoader)


# In[ ]:


# Save the FGSM model
torch.save(fgsm_model, "./cifar10_trained_fgsm_model")


# ### PGD Adversarial Training

# In[ ]:


pgd_model = get_adversarially_trained_model(
    "PGD", epsilon=(8 / 255), alpha=(2 / 255), iterations=7
)


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 0.1, 0.2, 0.35, 0.55]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        pgd_model,
        testSetLoader,
        "FGSM",
        attacks["FGSM"],
        epsilon=epsilon,
    )


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 4 / 255, 0.1, 0.3]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        pgd_model,
        testSetLoader,
        "PGD",
        attacks["PGD"],
        epsilon=epsilon,
        alpha=(2 / 255),
        iterations=7,
    )


# In[ ]:


# Make sure to test the final accuracy of the model
clean_test.test_trained_model(pgd_model, testSetLoader)


# In[ ]:


torch.save(pgd_model, "./cifar10_trained_pgd_model")


# ## Loading a good (saved) model

# In[ ]:


pgd_model = torch.load("./models_data/cifar10_pgd_model_200_epochs")
pgd_model.eval()

clean_test.test_trained_model(pgd_model, testSetLoader)


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 0.1, 0.2, 0.35, 0.55]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        pgd_model,
        testSetLoader,
        "FGSM",
        attacks["FGSM"],
        epsilon=epsilon,
    )


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 4 / 255, 0.1, 0.3]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        pgd_model,
        testSetLoader,
        "PGD",
        attacks["PGD"],
        epsilon=epsilon,
        alpha=(2 / 255),
        iterations=7,
    )


# In[ ]:


attacking.attack_model(
    pgd_model,
    testSetLoader,
    "PGD",
    attacks["PGD"],
    epsilon=(4 / 255),
    alpha=(2 / 255),
    iterations=20,
)


# ## Interpolated Adversarial Training (IAT)

# In[7]:


iat_pgd_model = get_interpolated_adversarially_trained_model(
    "PGD", epsilon=(8 / 255), alpha=(2 / 255), iterations=7
)


# In[10]:


# iat_pgd_model = torch.load("models_data/cifar_trained_iat_model_200_epochs")


# In[11]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 0.1, 0.2, 0.35, 0.55]


# In[12]:


for epsilon in epsilons:
    attacking.attack_model(
        iat_pgd_model,
        testSetLoader,
        "FGSM",
        attacks["FGSM"],
        epsilon=epsilon,
    )


# In[13]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 4 / 255, 0.1, 0.3]


# In[14]:


for epsilon in epsilons:
    attacking.attack_model(
        iat_pgd_model,
        testSetLoader,
        "PGD",
        attacks["PGD"],
        epsilon=epsilon,
        alpha=(2 / 255),
        iterations=7,
    )


# In[15]:


attacking.attack_model(
    iat_pgd_model,
    testSetLoader,
    "PGD",
    attacks["PGD"],
    epsilon=(4 / 255),
    alpha=(2 / 255),
    iterations=20,
)


# In[ ]:


torch.save(pgd_model, "./cifar10_trained_iat_model")


# ## Interpolated Adversarial Training (with Jacobian Regularization)

# In[16]:


from jacobian import JacobianReg

import defenses.iat as iat


def get_jacobian_interpolated_adversarially_trained_model(attack, **kwargs):
    # Helps adjust learning rate for better results
    def adjust_learning_rate(optimizer, epoch, learning_rate):
        actual_learning_rate = learning_rate
        if epoch >= 150:
            actual_learning_rate = 0.01
        if epoch >= 200:
            actual_learning_rate = 0.001
        for param_group in optimizer.param_groups:
            param_group["lr"] = actual_learning_rate

    # Various training parameters
    epochs = 200
    learning_rate = 0.1

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = resnet.ResNet18()
    model = model.to(device)
    model = nn.DataParallel(model)

    # Jacobian regularization
    jacobian_reg = JacobianReg()
    jacobian_reg_lambda = 0.01

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

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

    # Get the attack
    attack_function = attacks[attack]

    print("Training the model using adversarial examples...")

    # Use a pretty progress bar to show updates
    for epoch in tnrange(epochs, desc="Adversarial Training Progress"):
        adjust_learning_rate(optimizer, epoch, learning_rate)

        for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
            # Cast to proper tensors
            images, labels = images.to(device), labels.to(device)

            # Use manifold mixup to modify the data
            (
                benign_mix_images,
                benign_mix_labels_a,
                benign_mix_labels_b,
                benign_mix_lamda,
            ) = iat.mix_inputs(1, images, labels)

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

            # Predictions are regularization
            benign_mix_images.requires_grad = True
            adversarial_mix_images.requires_grad = True

            optimizer.zero_grad()

            # Predict and calculate benign loss
            benign_logits = model(benign_mix_images)

            benign_loss = iat.mixup_loss_function(
                loss_function,
                benign_mix_lamda,
                benign_logits,
                benign_mix_labels_a,
                benign_mix_labels_b,
            )

            # Introduce Jacobian regularization
            jacobian_reg_loss = jacobian_reg(benign_mix_images, benign_logits)

            # Total benign loss
            benign_loss = benign_loss + jacobian_reg_lambda * jacobian_reg_loss

            # Predict and calculate adversarial loss
            adversarial_logits = model(adversarial_mix_images)
            adversarial_loss = iat.mixup_loss_function(
                loss_function,
                adversarial_mix_lamda,
                adversarial_logits,
                adversarial_mix_labels_a,
                adversarial_mix_labels_b,
            )

            # Introduce Jacobian regularization
            jacobian_reg_loss = jacobian_reg(adversarial_mix_images, adversarial_logits)

            # Total adversarial loss
            adversarial_loss = (
                adversarial_loss + jacobian_reg_lambda * jacobian_reg_loss
            )

            # Take average of the two losses
            loss = (benign_loss + adversarial_loss) / 2

            # Gradient descent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

    print("... done!")

    # Return the trained model
    return model


# In[17]:


iat_jacobian_pgd_model = get_jacobian_interpolated_adversarially_trained_model(
    "PGD", epsilon=(8 / 255), alpha=(2 / 255), iterations=7
)

iat_jacobian_pgd_model.eval()


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 0.1, 0.2, 0.35, 0.55]


# In[ ]:


# Run test for each epsilon
for epsilon in epsilons:
    attacking.attack_model(
        iat_jacobian_pgd_model,
        testSetLoader,
        "FGSM",
        attacks["FGSM"],
        epsilon=epsilon,
    )


# In[ ]:


# Several values to use for the epsilons
epsilons = [0, 0.05, 4 / 255, 0.1, 0.3]


# In[ ]:


for epsilon in epsilons:
    attacking.attack_model(
        iat_jacobian_pgd_model,
        testSetLoader,
        "PGD",
        attacks["PGD"],
        epsilon=epsilon,
        alpha=(2 / 255),
        iterations=7,
    )


# In[ ]:


clean_test.test_trained_model(iat_jacobian_pgd_model, testSetLoader)


# In[ ]:


torch.save(iat_jacobian_pgd_model, "./cifar10_trained_iat_jacobian_pgd_model")

