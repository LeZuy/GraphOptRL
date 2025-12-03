import torch
from envs.graph_env import GraphEnv
from models.gnn_encoder import GNNEncoder
from models.policy import Policy
from rl.reinforce import REINFORCE
from rl.buffer import Buffer

def train():
    n = 50
    k = 4
    T = 20
    episodes = 5000

    env = GraphEnv(n, k, T)

    gnn = GNNEncoder(in_dim=1, hidden=64, out_dim=64)
    policy = Policy(gnn, hidden_dim=64)

    agent = REINFORCE(policy, lr=1e-4, gamma=1.0)

    for ep in range(episodes):
        buffer = Buffer()
        A = env.reset()

        done = False
        while not done:
            action, logp_sum = policy.sample_action(A)
            A2, reward, done = env.step(action)

            buffer.states.append(A2)
            buffer.actions.append(action)
            buffer.logprobs.append(logp_sum)
            buffer.rewards.append(reward)

            A = A2

        agent.update(buffer)

        if ep % 50 == 0:
            print(f"Episode {ep} | Reward = {reward:.4f}")

if __name__ == "__main__":
    train()
