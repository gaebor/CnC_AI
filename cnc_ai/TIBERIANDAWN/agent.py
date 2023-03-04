from collections import defaultdict, deque

import torch
import numpy
from tqdm import trange

from cnc_ai.agent import AbstractAgent
from cnc_ai.TIBERIANDAWN.model import TD_GamePlay
from cnc_ai.nn import interflatten
from cnc_ai.common import dictmap, retrieve, plot_images


class NNAgent(AbstractAgent):
    def __init__(self, **model_params):
        self.device: str = 'cpu'
        self.dtype: torch.dtype = torch.float32
        self.model_params = model_params
        self.nn = TD_GamePlay(**model_params)
        self.nn.eval()
        self.optimizer = None
        self.optimizer_params = None

    def init_optimizer(self, **params):
        self.nn.train()
        self.optimizer_params = params
        self.optimizer = torch.optim.NAdam(self.nn.parameters(), **params)

    def __call__(self, **game_state_tensor):
        game_state_tensor = dictmap(game_state_tensor, self._to_device)
        action_parameters = self.nn(**game_state_tensor)
        self.plot_actions(*action_parameters)
        actions = interflatten(self.nn.actions.sample, *action_parameters)
        return tuple(action.cpu().numpy() for action in actions)

    def plot_actions(self, mouse_positional_params, action_logits):
        n_players = action_logits.shape[1]
        images = numpy.exp(retrieve(action_logits[0]))
        images /= images.sum(axis=1).sum(axis=1)[:, None, None]
        x = (torch.linspace(0, 1, 51)[:, None] * torch.ones(n_players)[None, :]).to(self.device)
        mouse_params = mouse_positional_params[0, :, 1]
        mouse_x = retrieve(torch.exp(-self.nn.actions.mouse_x.surprise(mouse_params[:, :2], x)))
        mouse_y = retrieve(torch.exp(-self.nn.actions.mouse_y.surprise(mouse_params[:, 2:], x)))
        mouse_position = mouse_x.T[:, :, None] * mouse_y.T[:, None, :]
        plot_images(images, mouse_position)

    def learn(self, game_state_tensors, actions, rewards, n=1, time_window=200):
        game_state_tensors = dictmap(game_state_tensors, self._to_device)
        actions = tuple(map(self._to_device, actions))
        print(rewards)
        rewards = self._to_device(rewards)
        progress_bar = trange(time_window, time_window + n, leave=True)
        for time_step in progress_bar:
            self.nn.reset()
            for i in trange(0, actions[0].shape[0], time_step, leave=False):
                self.optimizer.zero_grad()
                action_parameters = self.nn(
                    **dictmap(game_state_tensors, lambda t: t[i : i + time_step])
                )
                self.plot_actions(*action_parameters)
                actions_surprise = interflatten(
                    self.nn.actions.surprise,
                    *action_parameters,
                    *map(lambda t: t[i : i + time_step], actions),
                )
                games_surprise = actions_surprise.mean(axis=0)
                current_performance = (
                    '[' + ', '.join(map('{:.3f}'.format, retrieve(games_surprise))) + ']'
                )
                progress_bar.set_description(current_performance)
                objective = games_surprise.dot(rewards.to(games_surprise.dtype))
                objective.backward()
                self.optimizer.step()

    def save(self, path):
        self.nn.reset()
        parameters = {
            'class': type(self.nn).__name__,
            'init': self.model_params,
            'state_dict': self.nn.state_dict(),
            'optimizer_init': self.optimizer_params,
            'optimizer_class': type(self.optimizer).__name__,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(parameters, path)

    def _to_device(self, x):
        return torch.tensor(
            x,
            device=self.device,
            dtype={
                numpy.dtype('bool'): torch.bool,
                numpy.dtype('float32'): self.dtype,
                numpy.dtype('float64'): self.dtype,
                numpy.dtype('int32'): torch.int,
                numpy.dtype('int64'): torch.int,
            }[x.dtype],
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

        agent.init_optimizer(**parameters['optimizer_init'])
        assert type(agent.optimizer).__name__ == parameters['optimizer_class']
        agent.optimizer.load_state_dict(parameters['optimizer_state_dict'])
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
