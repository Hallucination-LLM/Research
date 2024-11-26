import logging
import os
from typing import Optional

import torch
from golemai.config import LOGGER_LEVEL
from golemai.enums import AutencoderReconKeys
from golemai.io.file_ops import load_json, save_json
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_layer_size: int = 128,
        encoding_dim: int = 64,
    ):
        """
        Initializes the Encoder.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_layer_size (int): Size of the hidden layers.
            encoding_dim (int): Dimension of the encoded representation.
        """
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(
        self,
        encoding_dim: int = 64,
        hidden_layer_size: int = 128,
        input_dim: int = 1024,
    ):
        """
        Initializes the Decoder.

        Args:
            encoding_dim (int): Dimension of the encoded representation.
            hidden_layer_size (int): Size of the hidden layers.
            input_dim (int): Dimension of the output data.
        """
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoding_dim, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_layer_size, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        encoding_dim: int = 64,
        hidden_layer_size: int = 128,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initializes the Autoencoder by combining Encoder and Decoder.

        Args:
            input_dim (int): Dimension of the input data.
            encoding_dim (int): Dimension of the encoded representation.
            hidden_layer_size (int): Size of the hidden layers.
            device (torch.device): Device to run the model on.
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_layer_size, encoding_dim)
        self.decoder = Decoder(encoding_dim, hidden_layer_size, input_dim)
        self.device = device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def train_ae(
        self,
        x: torch.Tensor,
        val_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 5,
        model_sub_dir: Optional[str] = os.path.join("models", "autoencoder"),
    ) -> None:
        """
        Train the autoencoder with early stopping.

        Args:
            x (torch.Tensor): Input data to train on.
            val_split (float): Fraction of the data to use for validation.
            epochs (int): Number of epochs to train.
            batch_size (int): Size of the batches.
            learning_rate (float): Learning rate for the optimizer.
            patience (int): Patience for early stopping.
            model_sub_dir (str): Subdirectory to save the model and reconstruction errors.
        """
        logger.info(
            f"train_ae: {x.shape = }, {val_split = }, {epochs = }, {batch_size = }, {learning_rate = }, {patience = }, {model_sub_dir = }"
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        train_x, val_x = train_test_split(
            x, test_size=val_split, random_state=42
        )

        train_dataloader = DataLoader(
            train_x, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_x, batch_size=batch_size)

        best_loss = float("inf")
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch in train_dataloader:
                inputs = batch.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            self.eval()
            val_loss = 0
            with torch.inference_mode():
                for batch in val_dataloader:
                    inputs = batch.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, inputs)
                    val_loss += loss.item()

            val_loss /= len(val_dataloader)
            logger.debug(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = self.state_dict()
                epochs_no_improve = 0
                logger.debug(
                    f"New best model found at epoch {epoch + 1} with Val Loss: {val_loss:.6f}"
                )
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.debug(
                    "Early stopping due to no improvement in validation loss."
                )
                break

        min_recon, max_recon = self._find_min_max_recon(train_x)

        if model_sub_dir and best_model_state is not None:
            self._save_model_and_reconstruction_errors(
                best_model_state, min_recon, max_recon, model_sub_dir
            )

    def _calculate_reconstruction_error(
        self, data: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction error for the given data using the trained autoencoder.

        Args:
            data (torch.Tensor): Data to calculate the reconstruction error.

        Returns:
            torch.Tensor: Reconstruction errors.
        """
        logger.debug(
            f"Calculating reconstruction error for data of shape {data.shape}"
        )
        self.eval()
        with torch.inference_mode():
            inputs = data.to(self.device)
            outputs = self(inputs)
            loss = nn.MSELoss(reduction="none")
            reconstruction_error = loss(outputs, inputs).mean(dim=1)
        return reconstruction_error

    def _find_min_max_recon(self, train_data: torch.Tensor) -> float:
        """
        Find the optimal threshold for classification based on the maximum reconstruction error in the training data.

        Args:
            train_data (torch.Tensor): Training data to calculate reconstruction errors.

        Returns:
            float: Minimum reconstruction error.
            float: Maximum reconstruction error.
        """
        logger.debug(
            "Finding optimal threshold based on maximum reconstruction error in training data..."
        )
        reconstruction_errors = self._calculate_reconstruction_error(train_data)
        max_recon = torch.max(reconstruction_errors).item()
        min_recon = torch.min(reconstruction_errors).item()
        mean_recon = torch.mean(reconstruction_errors).item()
        median_recon = torch.median(reconstruction_errors).item()
        logger.debug(
            f"Maximum reconstruction error: {max_recon}, Minimum reconstruction error: {min_recon}, \nMean reconstruction error: {mean_recon}, Median reconstruction error: {median_recon}"
        )
        return min_recon, max_recon

    def _save_model_and_reconstruction_errors(
        self,
        model_state_dict: dict,
        min_recon: float,
        max_recon: float,
        model_sub_dir: str,
    ) -> None:
        """
        Save the model weights and reconstruction errors in a subfolder.

        Args:
            model_state_dict (dict): The state dictionary of the model.
            min_recon (float): Minimum reconstruction error.
            max_recon (float): Maximum reconstruction error.
            model_sub_dir (str): Subfolder to save the model and reconstruction errors.
        """
        os.makedirs(model_sub_dir, exist_ok=True)

        model_name = os.path.basename(model_sub_dir)

        model_save_path = os.path.join(model_sub_dir, f"{model_name}.pth")
        recon_save_path = os.path.join(model_sub_dir, f"{model_name}.json")

        torch.save(model_state_dict, model_save_path)
        print(f"Model weights saved to {model_save_path}")

        data = {
            AutencoderReconKeys.MIN_RECON_ERROR.value: min_recon,
            AutencoderReconKeys.MAX_RECON_ERROR.value: max_recon,
        }

        save_json(data, recon_save_path)

        print(f"Reconstruction errors saved to {recon_save_path}")

    @staticmethod
    def load_model(
        model_sub_dir: str,
        input_dim: int = 1024,
        encoding_dim: int = 64,
        hidden_layer_size: int = 128,
        device: torch.device = torch.device("cuda"),
    ) -> tuple:
        """
        Load the model and thresholds from the saved files.

        Args:
            model_sub_dir (str): Subfolder containing the model and thresholds.
            input_dim (int): Dimension of the input data.
            encoding_dim (int): Dimension of the encoded representation.
            hidden_layer_size (int): Size of the hidden layers.
            device (torch.device): Device to load the model onto.

        Returns:
            model (Autoencoder): Loaded Autoencoder model.
            min_recon (float): Minimum reconstruction error.
            max_recon (float): Maximum reconstruction error.
        """
        logger.debug(f"Loading model and thresholds from {model_sub_dir}")

        model_name = os.path.basename(model_sub_dir)

        model_load_dir = os.path.join(model_sub_dir, f"{model_name}.pth")
        recon_load_dir = os.path.join(model_sub_dir, f"{model_name}.json")

        model = Autoencoder(
            input_dim=input_dim,
            encoding_dim=encoding_dim,
            hidden_layer_size=hidden_layer_size,
            device=device,
        )

        model.load_state_dict(torch.load(model_load_dir, map_location=device))
        model.to(device)
        logger.debug(f"Model loaded from {model_load_dir}")

        recon_data = load_json(recon_load_dir)

        min_recon = recon_data[AutencoderReconKeys.MIN_RECON_ERROR.value]
        max_recon = recon_data[AutencoderReconKeys.MAX_RECON_ERROR.value]
        logger.debug(f"Reconstruction errors loaded from {recon_load_dir}")

        return model, min_recon, max_recon

    def predict(
        self,
        x: torch.Tensor,
        threshold: float,
        anomaly_is_positive: bool = True,
    ) -> torch.Tensor:
        """
        Predict anomalies in the given data based on the reconstruction error and threshold.

        Args:
            x (torch.Tensor): Data to predict anomalies.
            threshold (float): Threshold for classification.
            anomaly_is_positive (bool): Whether anomaly is considered positive or negative.

        Returns:
            torch.Tensor: Predictions (by default - 1 for anomaly, 0 for normal).
        """
        logger.debug(f"Predicting anomalies in data of shape {x.shape}")
        self.eval()
        reconstruction_errors = self._calculate_reconstruction_error(x)
        predictions = (
            (reconstruction_errors > threshold).int()
            if anomaly_is_positive
            else (reconstruction_errors <= threshold).int()
        )
        logger.debug(f"Predictions calculated.")
        return predictions

    def predict_proba(
        self,
        x: torch.Tensor,
        min_recon: float,
        max_recon: float,
        anomaly_is_positive: bool = True,
    ) -> torch.Tensor:
        """
        Predict anomaly probabilities in the given data based on the reconstruction error and thresholds.

        Args:
            x (torch.Tensor): Data to predict anomaly probabilities.
            min_recon (float): Minimum reconstruction error.
            max_recon (float): Maximum reconstruction error.
            anomaly_is_positive (bool): Whether anomaly is considered positive or negative.

        Returns:
            torch.Tensor: Anomaly probabilities.
        """
        logger.info(
            f"predict_proba: {x.shape = }, {min_recon = }, {max_recon = }, {anomaly_is_positive = }"
        )
        self.eval()
        reconstruction_errors = self._calculate_reconstruction_error(x)
        extended_max_recon = max_recon + (max_recon - min_recon)
        probabilities = 1 - (reconstruction_errors - min_recon) / (
            extended_max_recon - min_recon
        )
        probabilities = torch.clamp(probabilities, 0.0, 1.0)

        if anomaly_is_positive:
            probabilities = 1 - probabilities

        logger.debug("Probabilities calculated.")
        return probabilities
