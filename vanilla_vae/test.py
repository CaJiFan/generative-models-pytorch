# test.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Import from model.py
from model import VAE # We only need the VAE class for inference/generation

# Hyperparameters (should match the trained model, especially latent_dim and input_dim)
INPUT_DIM = 28 * 28
HIDDEN_DIM = 400 # Not strictly needed for loading state_dict, but good for consistency if building model from scratch
LATENT_DIM = 20   # IMPORTANT: Change to 2 if you trained with latent_dim=2 and want to visualize
MODEL_LOAD_PATH = "./vae_mnist.pth"
GENERATED_SAMPLES_DIR = "./generated_samples"
LATENT_VIS_DIR = "./latent_visualizations"

def test_and_visualize_vae():
    # Create output directories if they don't exist
    if not os.path.exists(GENERATED_SAMPLES_DIR):
        os.makedirs(GENERATED_SAMPLES_DIR)
    if LATENT_DIM == 2 and not os.path.exists(LATENT_VIS_DIR):
        os.makedirs(LATENT_VIS_DIR)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_LOAD_PATH}. Please train the model first using train.py.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval() # Set model to evaluation mode

    # 1. Generate some samples from the latent space
    print("Generating samples from latent space...")
    with torch.no_grad():
        # Sample z from the prior (standard normal)
        num_generated_samples = 64
        sample_z = torch.randn(num_generated_samples, LATENT_DIM).to(device)
        generated_images = model.decode(sample_z).cpu()
        torchvision.utils.save_image(generated_images.view(num_generated_samples, 1, 28, 28),
                                     os.path.join(GENERATED_SAMPLES_DIR, 'generated_samples.png'))
        print(f"Generated samples saved to {os.path.join(GENERATED_SAMPLES_DIR, 'generated_samples.png')}")

    # 2. Visualize latent space (if latent_dim is 2)
    if LATENT_DIM == 2:
        print("Visualizing 2D latent space (if LATENT_DIM is 2)...")
        # MNIST Test Dataset for visualization
        test_dataset = torchvision.datasets.MNIST(root='./data',
                                                  train=False,
                                                  transform=transforms.ToTensor())
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1000, # Larger batch for faster processing
                                 shuffle=False)

        latent_coords_all = []
        labels_all = []
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(device)
                mu, _ = model.encode(data.view(-1, INPUT_DIM))
                latent_coords_all.append(mu.cpu())
                labels_all.append(labels.cpu())

        latent_coords_all = torch.cat(latent_coords_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(latent_coords_all[:, 0].numpy(), latent_coords_all[:, 1].numpy(), c=labels_all.numpy(), cmap='viridis', alpha=0.7, s=10)
        plt.colorbar(scatter, label='Digit Label')
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.title("MNIST Test Set Latent Space (Mean Vectors)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(LATENT_VIS_DIR, "latent_space_visualization.png"))
        print(f"Latent space visualization saved to {os.path.join(LATENT_VIS_DIR, 'latent_space_visualization.png')}")
        # plt.show() # Uncomment to display the plot directly
    elif LATENT_DIM != 2:
        print(f"Latent space visualization is only available if LATENT_DIM is 2. Current LATENT_DIM is {LATENT_DIM}.")

    # (Optional) Evaluate test set reconstruction loss
    # You can add this if needed, similar to how it was in the original combined script's training loop
    # test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    # test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loss_total = 0
    # with torch.no_grad():
    #     for data, _ in test_loader:
    #         data = data.to(device)
    #         reconstructed_data, mu, logvar = model(data)
    #         test_loss_total += loss_function(reconstructed_data, data, mu, logvar, INPUT_DIM).item()
    # avg_test_loss = test_loss_total / len(test_loader.dataset)
    # print(f"====> Average Test set reconstruction loss: {avg_test_loss:.4f}")


if __name__ == '__main__':
    # You might want to set LATENT_DIM here or via command-line arguments
    # if you often switch between values for testing.
    # For example, to test a model trained with LATENT_DIM=2:
    # LATENT_DIM = 2
    test_and_visualize_vae()