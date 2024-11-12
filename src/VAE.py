
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import random
import ast
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def VAE_similarity(df_robot: pd.DataFrame, df_animal: pd.DataFrame) -> list:
    df_robot = df_robot.drop(columns=['frame'], errors='ignore')
    df_animal = df_animal.drop(columns=['frame'], errors='ignore')
    df_animal = df_animal.drop(columns=['robot_index'], errors='ignore')

    if 'robot_index' not in df_robot.columns:
        raise ValueError("df_robot DataFrame must contain 'robot_index' column.")

    similarities = []
    animal_vectors = df_animal.values
    for robot_index, group_robot in df_robot.groupby('robot_index'):
        robot_vectors = group_robot.drop(columns=['robot_index']).values
        distances = cdist(robot_vectors, animal_vectors, metric='euclidean')
        avg_distance = distances.mean()
        similarities.append(avg_distance)

    print('VAE_similarity',similarities)
    return similarities


def infer_on_csv(df: pd.DataFrame) -> pd.DataFrame:
    scale_data = df.copy()

    for column in ['head', 'middle', 'rear', 'left_front', 'right_front', 'left_hind', 'right_hind']:
        scale_data[column] = scale_data[column].apply(parse_tuple_string)

    coordinates_robot_list = []
    robot_indices = []
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
        robot_indices.append(frame.get('robot_index'))
    coords_array = KeypointsDataset(coordinates_robot_list)

    embedding_vectors = []
    # Load the model
    latent_dim = 50
    input_dim = 14 * 5
    vae_loaded = VAE(input_dim, latent_dim)
    model_save_path = './src/model/vae_model.pth'
    vae_loaded.load_state_dict(torch.load(model_save_path, weights_only=True)) 
    vae_loaded.eval()

    with torch.no_grad():
        for j in range(len(coords_array)):
            sample_data, mean, std = coords_array[j]
            _, _, _, z_animal = vae_loaded(sample_data.unsqueeze(0))
            embedding_vector = z_animal.squeeze().numpy()

            embedding_vectors.append([j, robot_indices[j]] + embedding_vector.tolist())

    latent_df = pd.DataFrame(embedding_vectors, columns=['frame', 'robot_index'] + [f'latent_{k}' for k in range(latent_dim)])
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

def plot_fitness(fitnesses_all,distance_all, animal_similarity_all):
    max_distances = [np.min(distance) for distance in distance_all]
    max_similarities = [np.min(similarity) for similarity in animal_similarity_all]
    max_fitnesses = [np.min(fitness) for fitness in fitnesses_all]
    print("Max distances:", max_distances)
    print("Max similarities:", max_similarities)
    print("Max fitnesses:", max_fitnesses)
    iterations = np.arange(1, len(distance_all) + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.subplots_adjust(hspace=1)

    axs[0].plot(iterations, max_distances, label="Distance", color='blue', marker='o')
    axs[0].set_title("Distance over Generations")
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Max Distance")
    axs[0].legend()

    axs[1].plot(iterations, max_similarities, label="Animal Similarity", color='green', marker='x')
    axs[1].set_title("Animal Similarity over Generations")
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Max Animal Similarity")
    axs[1].legend()

    axs[2].plot(iterations, max_fitnesses, label="Sum of Distance & Animal Similarity", color='red', marker='s')
    axs[2].set_title("Fitness over Generations")
    axs[2].set_xlabel("Generation")
    axs[2].set_ylabel("Max Fitness Value")
    axs[2].legend()

    # plt.tight_layout()
    plt.show()


