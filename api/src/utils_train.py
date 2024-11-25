import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

def to_dataloader(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor = None,
    y_test: torch.Tensor = None,
    batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if X_test is None or y_test is None:
        return train_dl, None

    test_ds = TensorDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    return train_dl, test_dl

def count_correct(
    y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(y_pred, dim=1)
    return (y_true == preds).float().sum()

def validate(
    model: nn.Module,
    loss_fn: torch.nn.CrossEntropyLoss,
    dataloader: DataLoader,
    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:

    loss = 0
    correct = 0
    all_samples = len(dataloader.dataset)

    for i, (X_batch, y_batch) in enumerate(dataloader):

        X_batch, y_batch = X_batch.to(device), y_batch.unsqueeze(1).type(torch.FloatTensor).to(device)

        y_pred = model(X_batch)

        loss += loss_fn(y_pred, y_batch)
        correct += count_correct(y_pred, y_batch)

        # print(f"Batch {i+1}: Loss: {loss / len(y_pred)}, Accuracy: {correct / len(y_pred)}")
        
    return (loss / all_samples).item(), (correct / all_samples).item()

def fit(
    model: nn.Module, optimiser: optim.Optimizer,
    loss_fn: torch.nn.CrossEntropyLoss, 
    train_dl: DataLoader,
    val_dl: DataLoader,
    epochs: int,
    test_dl: DataLoader = None,
    print_metrics: str = True,
    device: str = 'cuda'):

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        for X_batch, y_batch in train_dl:

            X_batch, y_batch = X_batch.to(device), y_batch.unsqueeze(1).type(torch.FloatTensor).to(device)
            y_pred = model(X_batch) # Uzyskanie pseudoprawdopodobieństw dla próbek z minibatcha

            # print(f"y_true_device: {y_batch.device}, y_pred_device: {y_pred.device}")

            loss = loss_fn(y_pred, y_batch) # Policzenie funkcji straty
            loss.backward() # Wsteczna propagacja z wyniku funkcji straty - policzenie gradientów i zapisanie ich w tensorach (parametrach)
            optimiser.step() # Aktualizacja parametrów modelu przez optymalizator na podstawie gradientów zapisanych w tensorach (parametrach) oraz lr
            optimiser.zero_grad() # Wyzerowanie gradientów w modelu, alternatywnie można wywołać percepron.zero_grad()

        model.eval() # Przełączenie na tryb ewaluacji modelu - istotne dla takich warstw jak Dropuot czy BatchNorm
        with torch.no_grad():  # Wstrzymujemy przeliczanie i śledzenie gradientów dla tensorów - w procesie ewaluacji modelu nie chcemy zmian w gradientach
            train_loss, train_acc = validate(model, loss_fn, train_dl)
            val_loss, val_acc = validate(model, loss_fn, val_dl)
            if test_dl is not None:
                test_loss, test_acc = validate(model, loss_fn, test_dl)

        for metric, value in zip(
            ["train_loss", "train_acc", "val_loss", "val_acc"], 
            [train_loss, train_acc, val_loss, val_acc]
        ):
            metrics[metric].append(value)

        if test_dl is not None:
            metrics["test_loss"].append(test_loss)
            metrics["test_acc"].append(test_acc)

        if print_metrics:
            print(
                f"Epoch {epoch}: "
                f"train loss = {train_loss:.3f} (acc: {train_acc:.3f}), "
                f"validation loss = {val_loss:.3f} (acc: {val_acc:.3f})"
            )
            
    model.eval()
    return metrics
