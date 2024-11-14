import torch
from torch import nn

class MLP(nn.Module):

    def __init__(self, input_dim: int = 3584, hidden_dim: int = 128, output_dim: int = 1):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, mode: str = 'train'):
        x = self.fc1(x)
        x = self.batch_norm(x)
        x = self.elu(x)

        if mode == 'train':
            x = self.dropout(x)

        x = self.fc2(x)
        return x
    
    def predict(self, x, th: float = 0.5):
        return self.forward(x, mode='eval')[0].detach().cpu().item() > th
    
def load_detector(weights_path: str, device: str = 'cuda') -> nn.Module:

    model = MLP()
    model.to(device)
    model.load_state_dict(torch.load(weights_path))

    return model