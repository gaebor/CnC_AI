import torch

from cnc_ai.agent import AbstractAgent
from cnc_ai.TIBERIANDAWN.model import TD_GamePlay
from cnc_ai.nn import interflatten, save
from cnc_ai.common import dictmap


class NNAgent(AbstractAgent):
    def __init__(self, device='cuda'):
        self.device = device
        self.nn = TD_GamePlay().to(device)
        self.optimizer = torch.optim.SGD(self.nn.parameters(), lr=0.01, weight_decay=1e-6)

    def __call__(self, **game_state_tensor):
        game_state_tensor = dictmap(game_state_tensor, self.to_device)
        action_parameters = self.nn(**game_state_tensor)
        actions = interflatten(self.nn.actions.sample, *action_parameters)
        return tuple(action.cpu().numpy() for action in actions)

    def learn(self, game_state_tensors, actions, rewards):
        game_state_tensors = dictmap(game_state_tensors, self.to_device)
        actions = map(self.to_device, actions)
        rewards = self.to_device(rewards).to(torch.float32)

        self.optimizer.zero_grad()
        action_parameters = self.nn(**game_state_tensors)
        log_prob = interflatten(self.nn.actions.evaluate, *action_parameters, *actions).sum(axis=0)
        (-log_prob.dot(rewards)).backward()
        self.optimizer.step()

    def save(self, path):
        self.nn.reset()
        save(self.nn, path)

    def to_device(self, x):
        return torch.tensor(x, device=self.device)

    @staticmethod
    def load(path):
        pass


class DummyAgent(AbstractAgent):
    def __call__(self, **inputs):
        return

    @staticmethod
    def load(path):
        return DummyAgent()


class SimpleAgent(AbstractAgent):
    def __call__(self, **inputs):
        return

    @staticmethod
    def load(path):
        return SimpleAgent()
