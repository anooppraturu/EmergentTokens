import torch
from torch import nn

class WordPooler(nn.Module):
    def __init__(self, max_len: int = 7, mode: str = 'scalar'):
        super().__init__()
        self.max_len = max_len
        self.mode = mode

        if mode == 'scalar':
            self.alpha = nn.Parameter(torch.zeros(max_len))
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 7, d)
        if self.mode == 'scalar':
            w = self.alpha.view(1, self.max_len, 1)
            pooled = torch.sum(w*x, dim=1)
            return pooled
        

class WordClassifier(nn.Module):
    def __init__(self, d_model: int = 128, n_classes: int = 100):
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, v: torch.tensor):
        return self.linear(v)


class WordProbe(nn.Module):
    def __init__(self, d_model: int=128, n_classes: int = 100, max_len: int = 7, mode: str = 'scalar'):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.max_len = max_len
        self.mode = mode

        self.pooler = WordPooler(max_len=max_len, mode=mode)
        self.classifier = WordClassifier(d_model = d_model, n_classes = n_classes)

    def forward(self, X):
        # X (B, max_len, d_model)
        v = self.pooler(X)      # v (B, d_model)
        return self.classifier(v)