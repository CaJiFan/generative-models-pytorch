# train.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Import from model.py
from model import VAE, loss_function

# Hyperparameters
INPUT_DIM = 28 * 28  # MNIST image dimensions
HIDDEN_DIM = 400
LATENT_DIM = 20       # Can be changed, e.g., to 2 for 2D latent space visualization
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 20
MODEL_SAVE_PATH = "./vae_mnist.pth"
RECONSTRUCTION_DIR = "./reconstructions"

def train_vae():
    # Create reconstruction directory if it doesn't exist
    if not os.path.exists(RECONSTRUCTION_DIR):
        os.makedirs(RECONSTRUCTION_DIR)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # MNIST Dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    # Initialize model, optimizer
    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {NUM_EPOCHS} epochs...")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss_epoch = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Forward pass
            reconstructed_data, mu, logvar = model(data)

            # Compute loss
            loss = loss_function(reconstructed_data, data, mu, logvar, INPUT_DIM)
            train_loss_epoch += loss.item()

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(train_loader)}], Batch Loss: {loss.item()/len(data):.4f}")

        avg_train_loss = train_loss_epoch / len(train_loader.dataset)
        print(f"====> Epoch: {epoch+1} Average training loss: {avg_train_loss:.4f}")

        # Save some reconstructed images from the last batch of this epoch
        model.eval()
        with torch.no_grad():
            # Get the last batch data again (or a fixed sample)
            # For simplicity, we'll use the 'data' from the last training step of the epoch
            if data is not None:
                reconstructed_data, _, _ = model(data)
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1, 1, 28, 28),
                                      reconstructed_data.view(BATCH_SIZE, -1, 1, 28, 28)[:n]]) # Adjust view for batch
                torchvision.utils.save_image(comparison.cpu(),
                                             os.path.join(RECONSTRUCTION_DIR, f'reconstruction_epoch_{epoch+1}.png'), nrow=n)

    # Save the trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train_vae()