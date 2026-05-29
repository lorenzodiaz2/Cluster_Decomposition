from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class SimpleFeedforward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleFeedforward, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, 2)

        )
        self.decoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, input_size)
        )

        self.predictor = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization for weights
                nn.init.xavier_uniform_(module.weight)
                # Initialize biases to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)
        prediction = self.predictor(z)
        return z, x_hat, prediction

    def evaluate(self, x):
        with torch.no_grad():
            return self.forward(x)


class NN_Isa:

    def __init__(self, df: pd.DataFrame, feat_names: List[str], target_alg, baseline_alg, hidden_size=128):

        self.df = df
        # self.instance_col_name = instance_col_name
        self.feat_names = feat_names
        self.target_alg = target_alg
        self.baseline_alg = baseline_alg
        self.hidden_size = hidden_size
        self.df['improvement_perc'] = 100 * (df[self.baseline_alg] - df[self.target_alg]) / df[self.baseline_alg]
        self.df['improvement'] = ((self.df.improvement_perc - self.df.improvement_perc.min()) /
                                  (self.df.improvement_perc.max() - self.df.improvement_perc.min()))

        print('classes distribution', self.df.improvement[df.improvement > 0.5].shape[0],
              self.df.improvement[df.improvement < 0.5].shape[0])

        self.label = None
        self.train_dataloader, self.test_dataloader, self.X_tensor, self.Y_tensor = None, None, None, None

        self.model = SimpleFeedforward(len(self.feat_names), hidden_size)

        self.accuracy = None

    def run(self, batch_size=128, epochs=1000, alpha=0.1, lr=0.0001, train_test_ratio=0.5):
        self.train_dataloader, self.test_dataloader, self.X_tensor, self.Y_tensor = (
            self.prepare_datasets(self.df, self.feat_names, self.df.improvement, batch_size=batch_size, train_test_ratio=train_test_ratio))
        self.train_model(self.model, self.train_dataloader, self.test_dataloader, alpha=alpha, epochs=epochs, lr=lr)

        z, x_hat, prediction = self.model.evaluate(self.X_tensor)
        z = z.detach().cpu().numpy()
        label = self.Y_tensor.squeeze(1).detach().cpu().numpy()

        self.df['x_coord'] = z[:, 0]
        self.df['y_coord'] = z[:, 1]
        self.df['classification_label'] = label

        x_baseline = self.X_tensor[torch.where(self.Y_tensor > 0.5)[0]]
        _, _, accuracy_x_baseline = self.model.evaluate(x_baseline)

        x_target = self.X_tensor[torch.where(self.Y_tensor < 0.5)[0]]
        _, _, res_target = self.model.evaluate(x_target)

        self.accuracy = {'baseline': (torch.round(accuracy_x_baseline).sum() / accuracy_x_baseline.shape[0]).item(),
                         'target': ((1 - torch.round(res_target)).sum() / res_target.shape[0]).item()}
        print('Accuracy:', self.accuracy)

    def predict(self, x):
        if torch.Tensor != type(x):
            x = torch.Tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z, x_hat, prediction = self.model.evaluate(x)
        return z, x_hat, prediction

    def plot(self, identical_performance_threshold=0.05):
        th = identical_performance_threshold
        colors = ['b' if el <= 0.5 - th else ('k' if 0.5 - th < el <= 0.5 + th else 'r') for el in self.df['classification_label']]

        # colors_quadratic = (y_ - 0.5) ** 2
        # colors_quadratic = (colors_quadratic - colors_quadratic.min()) / (colors_quadratic.max() - colors_quadratic.min())
        mpl.rcParams['font.size'] = 30
        mpl.rcParams['figure.figsize'] = (20, 15)
        plt.scatter(self.df.x_coord, self.df.y_coord, marker="o", c=colors, s=20)
        plt.show()

    @staticmethod
    def prepare_datasets(df, features_names, y, batch_size, train_test_ratio):
        X = df[features_names].to_numpy(dtype=float)

        # normalisation
        for i in range(X.shape[1]):
            if np.max(X[:, i]) != np.min(X[:, i]):
                X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))

        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(y).unsqueeze(1)

        # Create dataloader
        dataset = TensorDataset(X_tensor, Y_tensor)
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_indices = train_dataset.indices
        train_labels = y[train_indices].round().astype(int)  # Only training labels

        # Calculate class weights using ONLY training labels
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]

        # Create sampler for training only
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler  # Balanced batches
        )

        test_indices = test_dataset.indices
        test_labels = y[test_indices].round().astype(int)  # Only testing labels

        # Calculate class weights using ONLY testing labels
        class_counts = np.bincount(test_labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[test_labels]

        # Create sampler for testing only
        test_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_size,
            sampler=test_sampler  # No sampler for test
        )
        print(f"Total samples: {len(dataset)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        return train_dataloader, test_dataloader, X_tensor, Y_tensor

    @staticmethod
    def loss_fun(prediction, target, x, x_batch, alpha=1):
        # return alpha * torch.mean((torch.abs(target - 0.5) + 0.1 * (target - 0.5)/torch.abs(target - 0.5)) * (prediction - target)**2) + torch.mean((x - x_batch)**2)
        loss = nn.BCELoss()
        return loss(prediction, target) + 0.0001 * torch.mean((x - x_batch) ** 2)

    # Example training loop structure
    def train_model(self, model, train_loader, test_loader, alpha, epochs, lr):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initialize checkpoint variables (memory only)
        best_test_error = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                z, x_hat, prediction = model(batch_X)
                loss = self.loss_fun(prediction, batch_Y, x_hat, batch_X, alpha=alpha)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Evaluation phase
            model.eval()
            test_loss = 0
            errors = []
            with torch.no_grad():
                for batch_X, batch_Y in test_loader:
                    z, x_hat, prediction = model(batch_X)
                    loss = self.loss_fun(prediction, batch_Y, x_hat, batch_X, alpha=alpha)
                    errors.append((torch.round(prediction) - torch.round(batch_Y)).abs().mean().item())
                    test_loss += loss.item()

            current_test_error = np.mean(errors)

            # Save best model in memory (no disk write)
            if current_test_error < best_test_error:
                best_test_error = current_test_error
                best_model_state = model.state_dict().copy()  # Deep copy to memory
                print(f"*** New best model at epoch {epoch + 1} with test error: {best_test_error:.4f} ***")

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, "
                    f"Test Loss: {test_loss / len(test_loader):.4f}    Error: {current_test_error:.4f}")

        # Load the best model from memory
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"\nTraining complete. Loaded best model from epoch with test error: {best_test_error:.4f}")

