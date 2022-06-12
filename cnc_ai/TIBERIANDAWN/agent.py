import torch
import numpy

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
        # dynamic_mask,
        # sidebar_mask,
        # StaticAssetName,
        # StaticShapeIndex,
        # AssetName,
        # ShapeIndex,
        # Owner,
        # Pips,
        # ControlGroup,
        # Cloak,
        # Continuous,
        # SidebarInfos,
        # SidebarAssetName,
        # SidebarContinuous,
        actions = [
            SimpleAgent.get_action_single(dictmap(inputs, lambda x: x[-1][player]))
            for player in range(inputs['AssetName'].shape[1])
        ]
        actions = tuple(map(numpy.array, zip(*actions)))
        return actions

    @staticmethod
    def get_action_single(inputs):
        unit_names = inputs['AssetName'][~inputs['dynamic_mask']]
        sidebar = inputs['SidebarAssetName'][~inputs['sidebar_mask']]
        if 63 in unit_names:
            mcv_index = list(unit_names).index(63)
            if mcv_index >= 0:
                MCV_features = inputs['Continuous'][mcv_index]
                return 2, *MCV_features[:2]
        return 0, 0.0, 0.0

    @staticmethod
    def load(path):
        return SimpleAgent()


def mix_actions(actions1, actions2, choices):
    for action1, action2 in zip(actions1, actions2):
        action1[choices] = action2[choices]
    return tuple(actions1)
