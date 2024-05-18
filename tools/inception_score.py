import torch
from torchvision.models import inception_v3
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy.stats import entropy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inception_score(images, batch_size=32, splits=10):
    # Set up the Inception v3 model
    model = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
    model.eval()
    model.to(device)

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create dataloader for the images
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=False)

    # Initialize lists to store activations and scores
    preds = []

    # Iterate through the dataloader to get predictions
    for batch in dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(transform(batch))
            preds.append(pred.softmax(dim=1).cpu().numpy())

    # Concatenate predictions
    preds = np.concatenate(preds, axis=0)

    # Calculate the Inception score
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    return np.mean(scores), np.std(scores)


if __name__ == '__main__':
    # Load the generated images (fake images)
    # fake_images should be a PyTorch dataset or a list of tensors
    fake_images = ...

    # Calculate the Inception score
    mean_score, std_score = inception_score(fake_images)

    print("Inception Score:", mean_score, "+/-", std_score)
