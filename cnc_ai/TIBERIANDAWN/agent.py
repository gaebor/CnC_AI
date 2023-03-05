from collections import defaultdict, deque

import torch
import numpy
from tqdm import trange

from cnc_ai.agent import AbstractAgent
from cnc_ai.TIBERIANDAWN.model import TD_GamePlay
from cnc_ai.common import dictmap, retrieve, plot_images, numpy_to_torch
from cnc_ai.TIBERIANDAWN.bridge import GameAction, GameState, concatenate_game_actions


class NNAgent(AbstractAgent):
    def __init__(self, **model_params):
        self.device: str = 'cpu'
        self.dtype: torch.dtype = torch.float32
        self.model_params = model_params
        self.nn = TD_GamePlay(**model_params)
        self.nn.eval()
        self.optimizer = None

    def init_optimizer(self, **params):
        self.nn.train()
        self.optimizer = torch.optim.SGD(self.nn.parameters(), **params)

    def __call__(self, game_state_tensor: GameState) -> GameAction:
        game_state_tensor = dictmap(game_state_tensor.__dict__, self._to_device)
        action_parameters = self.nn(**game_state_tensor)
        # self.plot_actions(*action_parameters)
        actions = self.nn.actions.sample(*action_parameters)
        return GameAction(*map(lambda t: t.cpu().numpy()[-1], actions))

    def plot_actions(self, mouse_positional_params, action_logits):
        n_players = action_logits.shape[1]
        x = (torch.linspace(0, 1, 51)[:, None] * torch.ones(n_players)[None, :]).to(self.device)
        button_probabilities, mouse_x, mouse_y = self.nn.actions.get_probabilities(
            mouse_positional_params[-1], action_logits[-1], x
        )
        button_probabilities = retrieve(button_probabilities)
        mouse_x = retrieve(mouse_x)
        mouse_y = retrieve(mouse_y)
        mouse_position = mouse_x.T[:, :, None] * mouse_y.T[:, None, :]
        plot_images(button_probabilities, mouse_position)

    def learn(self, game_state_tensors, actions, rewards, n=1, time_window=200):
        game_state_tensors = dictmap(game_state_tensors, self._to_device)
        actions = tuple(map(self._to_device, actions))
        rewards = self._to_device(rewards.astype(float))
        progress_bar = trange(time_window, time_window + n, leave=True)
        for time_step in progress_bar:
            self.nn.reset()
            for i in trange(0, actions[0].shape[0], time_step, leave=False):
                self.optimizer.zero_grad()
                action_parameters = self.nn(
                    **dictmap(game_state_tensors, lambda t: t[i : i + time_step])
                )
                self.plot_actions(*action_parameters)
                actions_surprise = self.nn.actions.surprise(
                    *action_parameters,
                    *map(lambda t: t[i : i + time_step], actions),
                )
                games_surprise = actions_surprise.mean(axis=0)
                objective = games_surprise.dot(rewards)
                objective.backward()
                self.optimizer.step()
                progress_bar.set_description(f'{retrieve(objective):.4g}')

    def save(self, path):
        self.nn.reset()
        parameters = {
            'class': type(self.nn).__name__,
            'init': self.model_params,
            'state_dict': self.nn.state_dict(),
        }
        torch.save(parameters, path)

    def _to_device(self, x):
        return torch.tensor(
            x,
            device=self.device,
            dtype=numpy_to_torch(x.dtype, self.dtype),
        )

    def to(self, device=None, dtype=None):
        self.nn.to(device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    @staticmethod
    def load(path, map_location='cpu'):
        parameters = torch.load(path, map_location=map_location)
        agent = NNAgent(**parameters['init'])
        if type(agent.nn).__name__ != parameters['class']:
            raise ValueError(
                f"Class to load ('{type(agent.nn).__name__}') is not compatible with "
                f"saved class ('{parameters['class']}')"
            )
        agent.nn.load_state_dict(parameters['state_dict'])
        agent.nn.eval()
        agent.optimizer = None
        return agent


class SimpleAgent(AbstractAgent):
    class Player:
        def __init__(self):
            self.color = None
            self.actions = deque()

        def get_action(self, inputs: GameState) -> GameAction:
            if len(self.actions) > 0:
                i, j, x, y = self.actions.pop()
                return self.render_action(i, j, len(inputs.SidebarAssetName), x, y)
            unit_names = inputs.AssetName[~inputs.dynamic_mask]
            sidebar = inputs.SidebarAssetName[~inputs.sidebar_mask]
            s = len(inputs.SidebarAssetName)
            if 63 in unit_names:
                mcv_index = list(unit_names).index(63)
                if self.color is None:
                    self.color = inputs.Owner[mcv_index]
                MCV_features = inputs.Continuous[mcv_index]
                # move mouse to position then click with left mouse button
                self.actions.append((0, 1, MCV_features[0], MCV_features[1]))
                self.actions.append((0, 2, 744.0, 744.0))
                # do nothing for now
                return self.render_action(0, 0, s, 744.0, 744.0)

            elif 72 not in unit_names and 73 not in unit_names:
                if 72 in sidebar:
                    nuke = list(sidebar).index(72)
                    progress = inputs.SidebarContinuous[nuke][0]
                    if progress == 0:
                        # start building
                        return self.render_action(nuke, 0, s, 744.0, 744.0)
                    elif progress == 1:
                        new_spot = self.find_new_spot(inputs)
                        # move mouse to position
                        # then start placement
                        # then place
                        self.actions.append((0, 1, *new_spot))
                        self.actions.append((nuke, 3, 744.0, 744.0))
                        self.actions.append((nuke, 4, 744.0, 744.0))
                        return self.render_action(0, 0, s, 744.0, 744.0)
            return self.render_action(0, 0, s, 744.0, 744.0)

        @staticmethod
        def render_action(i, j, len_sidebar, mouse_x, mouse_y):
            action_matrix = numpy.zeros((len_sidebar, 12), dtype=bool)
            action_matrix[i, j] = True
            return GameAction(button=action_matrix, mouse_x=mouse_x, mouse_y=mouse_y)

        def find_new_spot(self, inputs: GameState):
            unit_positions = inputs.Continuous[~inputs.dynamic_mask][:, :2]
            unit_names = inputs.AssetName[~inputs.dynamic_mask]
            CY_position = unit_positions[list(unit_names).index(39)]
            diff = numpy.random.randint(-3, 4, size=2) * 24
            return CY_position + diff

    def __init__(self):
        self.players = defaultdict(SimpleAgent.Player)

    def __call__(self, inputs: GameState) -> GameAction:
        n_players = inputs.AssetName.shape[1]
        actions = concatenate_game_actions(
            [
                self.players[player].get_action(inputs.take(-1, player))
                for player in range(n_players)
            ]
        )
        return actions

    def load(path):
        return SimpleAgent()
