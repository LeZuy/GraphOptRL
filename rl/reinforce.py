import torch

class REINFORCE:
    def __init__(self, policy, lr=3e-4, gamma=1.0):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma

    def update(self, buffer):
        # Compute returns
        returns = []
        G = 0
        for r in reversed(buffer.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for logp, Gt in zip(buffer.logprobs, returns):
            loss += -logp * Gt

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()