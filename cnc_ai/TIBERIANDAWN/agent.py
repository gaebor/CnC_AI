from collections import defaultdict, deque

import torch
import numpy

from cnc_ai.agent import AbstractAgent
from cnc_ai.TIBERIANDAWN.model import TD_GamePlay
from cnc_ai.nn import interflatten
from cnc_ai.common import dictmap, retrieve


class NNAgent(AbstractAgent):
    def __init__(self, **model_params):
        self.device = 'cpu'
        self.model_params = model_params
        self.nn = TD_GamePlay(**model_params)
        self.optimizer = None

    def init_optimizer(self, lr=1e-4, weight_decay=1e-10, **params):
        self.optimizer = torch.optim.SGD(
            self.nn.parameters(), lr=lr, weight_decay=weight_decay, **params
        )

    def __call__(self, **game_state_tensor):
        game_state_tensor = dictmap(game_state_tensor, self._to_device)
        action_parameters = self.nn(**game_state_tensor)
        actions = interflatten(self.nn.actions.sample, *action_parameters)
        return tuple(action.cpu().numpy() for action in actions)

    def learn(self, game_state_tensors, actions, rewards, n=1):
        game_state_tensors = dictmap(game_state_tensors, self._to_device)
        actions = tuple(map(self._to_device, actions))
        rewards = self._to_device(rewards)
        for _ in range(n):
            self.optimizer.zero_grad()
            action_parameters = self.nn(**game_state_tensors)
            actions_surprise = interflatten(self.nn.actions.surprise, *action_parameters, *actions)
            games_surprise = actions_surprise.mean(axis=0)
            objective = games_surprise.dot(rewards.to(games_surprise.dtype))
            objective.backward()
            self.optimizer.step()
            print(retrieve(games_surprise), end='\r')

    def save(self, path):
        self.nn.reset()
        parameters = {
            'class': type(self.nn).__name__,
            'init': self.model_params,
            'state_dict': self.nn.state_dict(),
        }
        torch.save(parameters, path)

    def _to_device(self, x):
        return torch.tensor(x, device=self.device)

    def to(self, device):
        self.device = device
        self.nn.to(device)

    @staticmethod
    def load(path):
        parameters = torch.load(path)
        agent = NNAgent(**parameters['init'])
        if type(agent.nn).__name__ != parameters['class']:
            raise ValueError(
                f"Class to load ('{type(agent.nn).__name__}') is not compatible with "
                f"saved class ('{parameters['class']}')"
            )
        agent.nn.load_state_dict(parameters['state_dict'])
        return agent


class SimpleAgent(AbstractAgent):
    class Player:
        def __init__(self):
            self.color = None
            self.actions = deque()

        def get_action(self, inputs):
            if len(self.actions) > 0:
                return self.actions.pop()
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
            unit_names = inputs['AssetName'][~inputs['dynamic_mask']]
            sidebar = inputs['SidebarAssetName'][~inputs['sidebar_mask']]
            len_sidebar = inputs['SidebarAssetName'].shape[0]
            if 63 in unit_names:
                mcv_index = list(unit_names).index(63)
                if self.color is None:
                    self.color = inputs['Owner'][mcv_index]
                MCV_features = inputs['Continuous'][mcv_index]
                # move mouse to position then click with left mouse button
                self.actions.append((0, 1, MCV_features[0], MCV_features[1]))
                self.actions.append((0, 2, 744.0, 744.0))
                # do nothing for now
                return 0, 0, 744.0, 744.0

            elif 72 not in unit_names and 73 not in unit_names:
                if 72 in sidebar:
                    nuke = list(sidebar).index(72)
                    progress = inputs['SidebarContinuous'][nuke][0]
                    if progress == 0:
                        # start building
                        return nuke, 0, 744.0, 744.0
                    elif progress == 1:
                        new_spot = self.find_new_spot(inputs)
                        # move mouse to position
                        # then start placement
                        # then place
                        self.actions.append((0, 1, *new_spot))
                        self.actions.append((nuke, 3, 744.0, 744.0))
                        self.actions.append((nuke, 4, 744.0, 744.0))
                        return 0, 0, 744.0, 744.0
            return 0, 0, 744.0, 744.0

        def find_new_spot(self, inputs):
            unit_positions = inputs['Continuous'][~inputs['dynamic_mask']][:, :2]
            unit_names = inputs['AssetName'][~inputs['dynamic_mask']]
            CY_position = unit_positions[list(unit_names).index(39)]
            diff = numpy.random.randint(-3, 4, size=2) * 24
            return CY_position + diff

    def __init__(self):
        self.players = defaultdict(SimpleAgent.Player)

    def __call__(self, **inputs):
        actions = [
            self.players[player].get_action(dictmap(inputs, lambda x: x[-1][player]))
            for player in range(inputs['AssetName'].shape[1])
        ]
        actions = tuple(map(numpy.array, zip(*actions)))
        return actions

    @staticmethod
    def load(path):
        return SimpleAgent()


def mix_actions(actions1, actions2, choices):
    for action1, action2 in zip(actions1, actions2):
        action1[choices] = action2[choices]
    return tuple(actions1)
