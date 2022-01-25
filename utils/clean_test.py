import torch
from tqdm.notebook import tqdm


# Define the `device` PyTorch will be running on, please hope it is CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"


# This method tests a given mode and prints the accuracy of that model
def test_trained_model(model, testSetLoader):
    correct = 0

    print("Testing the model...")

    # Use a pretty progress bar to show updates
    for j, (images, labels) in enumerate(tqdm(testSetLoader, desc="Testing Progress", leave=False)):
        # Cast to proper tensor
        images, labels = images.to(device), labels.to(device)

        # Predict
        logits = model(images)

        # The highest class represents the chosen class
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()

    print("... done! Accuracy: {}%".format(float(correct) * 100 / 10000))