import torch
from envs.graph_env import GraphEnv
from rl.reinforce import REINFORCE
from rl.buffer import Buffer
from models.policy import Policy   # TRANSFORMER POLICY


def train():
    n = 50          # number of nodes
    k = 4           # degree
    T = 40          # steps per episode
    episodes = 2000

    env = GraphEnv(n, k, T)

    policy = Policy(hidden_dim=128)

    agent = REINFORCE(policy, lr=3e-5, gamma=1.0)

    for ep in range(episodes):
        buffer = Buffer()
        G = env.reset()

        done = False
        total_reward = 0

        while not done:
            # Sample action from Transformer policy
            action, logp = policy.sample_action(G)

            # Environment step
            G_next, reward, done = env.step(action)

            buffer.states.append(G_next)
            buffer.actions.append(action)
            buffer.logprobs.append(logp)
            buffer.rewards.append(reward)

            total_reward += reward
            G = G_next

        agent.update(buffer)

        if ep % 20 == 0:
            print(f"Episode {ep} | Reward = {total_reward:.4f}")


if __name__ == "__main__":
    train()