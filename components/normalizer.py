import torch

class Normalizer:
    def __init__(self, size, eps=1e-8, device=None):
        self.size = size
        self.eps = eps
        self.device = device if device else torch.device("cpu")
        self.mean = torch.zeros(size, dtype=torch.float32, device=self.device)
        self.var = torch.ones(size, dtype=torch.float32, device=self.device)
        self.count = 0

    def update(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return (x - self.mean) / (torch.sqrt(self.var) + self.eps)
        
    def denormalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x * (torch.sqrt(self.var) + self.eps) + self.mean
