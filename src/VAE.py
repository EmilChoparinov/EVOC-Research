
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import random
import ast

def infer_on_csv(df: pd.DataFrame) -> pd.DataFrame:
    scale_data = df

    for column in ['head', 'middle', 'rear', 'left_front', 'right_front', 'left_hind', 'right_hind']:
        scale_data[column] = scale_data[column].apply(parse_tuple_string)

    coordinates_robot_list = []
    for index, frame in scale_data.iterrows():
        coordinates_2 = {
            'head': frame.get('head'),
            'middle': frame.get('middle'),
            'rear': frame.get('rear'),
            'left_front': frame.get('left_front'),
            'right_front': frame.get('right_front'),
            'left_hind': frame.get('left_hind'),
            'right_hind': frame.get('right_hind'),
        }
        coordinates_robot_list.append(coordinates_2)

    coords_array = KeypointsDataset(coordinates_robot_list)

    embedding_vectors = []
    # Load the model
    latent_dim = 50
    input_dim = 14 * 5
    vae_loaded = VAE(input_dim, latent_dim)
    model_save_path = './src/model/vae_model.pth'
    vae_loaded.load_state_dict(torch.load(model_save_path, weights_only=True))  # 设置 weights_only=True

    vae_loaded.eval()  # Set to evaluation mode
    with torch.no_grad():
        for j in range(len(coords_array)):
            sample_data, mean, std = coords_array[j]
            original_animal_keypoints = sample_data.numpy()
            # Get the embedding vector (latent representation) using trained VAE
            _, _, _, z_animal = vae_loaded(sample_data.unsqueeze(0))
            embedding_vector = z_animal.squeeze().numpy()

            # Append the frame index and the embedding vector to the list
            embedding_vectors.append([j] + embedding_vector.tolist())

    # Convert embedding vectors to DataFrame
    latent_df = pd.DataFrame(embedding_vectors, columns=['frame'] + [f'latent_{k}' for k in range(latent_dim)])
    print(latent_df.head(3))
    return latent_df



def parse_tuple_string(s):
    if isinstance(s, tuple):
        return s
    elif pd.notna(s):
        return ast.literal_eval(s)
    else:
        return (None, None)

# Function to parse parameters and extract keypoints
def extract_robot_keypoints(row):
    # Extract coordinates from the head and box
    coords = []
    for key in ['head', 'middle', 'rear', 'left_front', 'right_front', 'left_hind', 'right_hind']:
        if row[key] != (None, None):
            coords.append(row[key][0])  # X-coordinate
            coords.append(row[key][1])  # Y-coordinate

    return np.array(coords).flatten() if coords else np.zeros(14)  # 7 keypoints x 2 (x, y)

class KeypointsDataset(Dataset):
    def __init__(self, robot_data, frames_per_sample=5, shuffle_data=True):
        self.data = []
        self.means = []
        self.stds = []
        self.frames_per_sample = frames_per_sample
        if shuffle_data:
            random.shuffle(robot_data)
        all_coords = []
        for robot_keypoints in robot_data:
            coords = extract_robot_keypoints(robot_keypoints)
            if len(coords) > 0:
                all_coords.append(coords)
        all_coords = np.array(all_coords)
        self.means = np.mean(all_coords, axis=0)
        self.stds = np.std(all_coords, axis=0)

        for robot_keypoints in robot_data:
            coords = extract_robot_keypoints(robot_keypoints)
            if len(coords) > 0:
                normalized_coords = (coords - self.means) / self.stds
                self.data.append(normalized_coords)

        self.data = np.array(self.data)

    def __len__(self):
        return len(self.data) - self.frames_per_sample + 1

    def __getitem__(self, idx):
        frame_sequence = self.data[idx:idx + self.frames_per_sample]
        combined_frames = torch.tensor(frame_sequence, dtype=torch.float32).flatten()

        return combined_frames, self.means, self.stds


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5):
        super(VAE, self).__init__()
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, latent_dim)  # mean
        self.fc32 = nn.Linear(64, latent_dim)  # log variance

        # Decoder layers
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return self.fc31(h2), torch.clamp(self.fc32(h2), min=-5, max=5)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def vae_loss(recon_x, x, mu, logvar, beta=0.001):
    """Compute the variational lower bound loss.

            log p(x) >= E[ log p(x|z) ] + KL(q(x) | prior)
                            recon_loss        kl_loss
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_divergence

