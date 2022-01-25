import torch
import torch.nn as nn
from tqdm.notebook import tnrange, tqdm

# For loading model sanely
import os.path
import sys

# This here actually adds the path
sys.path.append("../../")
import models.lenet as lenet

# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Notebook will use PyTorch Device: " + device.upper())


# This method creates a new model and also trains it
def standard_training(
  trainSetLoader,
  load_if_available=False,
  load_path="../models_data/MNIST/mnist_standard"
):
    # Helps speed up operations
    scaler = torch.cuda.amp.GradScaler()

    # Various training parameters
    epochs = 20
    learning_rate = 0.01

    # Network parameters
    loss_function = nn.CrossEntropyLoss()
    model = lenet.LeNet5().to(device)
    model.train()

    # Consider using ADAM here as another gradient descent algorithm
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )

    # If a trained model already exists, give up the training part
    if load_if_available and os.path.isfile(load_path):
        print("Found already trained model...")

        model = torch.load(load_path)

        print("... loaded!")
    else:
        print("Training the model...")

        # Use a pretty progress bar to show updates
        for epoch in tnrange(epochs, desc="Training Progress"):
            for _, (images, labels) in enumerate(tqdm(trainSetLoader, desc="Batches")):
                # Cast to proper tensors
                images, labels = images.to(device), labels.to(device)

                # Clean the gradients
                optimizer.zero_grad()

                # Predict
                logits = model(images)

                # Calculate loss
                with torch.cuda.amp.autocast():
                    loss = loss_function(logits, labels)

                # Gradient descent
                scaler.scale(loss).backward()

                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()

        print("... done!")

    # Make sure the model is in eval mode before returning
    model.eval()

    return model