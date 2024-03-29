from subprocess import Popen
from os import mkdir
import ctypes
from random import choice
from datetime import datetime
import pickle

from tqdm import trange
import numpy

import tornado.web
import tornado.websocket
import tornado.ioloop

from torch import float16, float32

from cnc_ai.TIBERIANDAWN import cnc_structs
from cnc_ai.TIBERIANDAWN.agent import NNAgent, SimpleAgent
from cnc_ai.TIBERIANDAWN.bridge import (
    concatenate_game_states,
    concatenate_game_actions,
    encode_list,
    GameAction,
)

from cnc_ai.arg_utils import get_args


class GameHandler(tornado.websocket.WebSocketHandler):
    def on_message(self, message):
        if len(message) == 1:
            self.loser_mask = ctypes.c_ubyte.from_buffer_copy(message).value
            self.close()
        else:
            # recieved the current game state
            offset = 0
            self.game_states.append([])
            for _ in range(len(self.players)):
                self.game_states[-1].append(cnc_structs.convert_to_np(message[offset:]))
                offset += cnc_structs.get_game_state_size(message[offset:])

            if len(self.game_states) >= GameHandler.end_limit:
                self.close()
                return

            # sync games
            if (GameHandler.n_games == len(GameHandler.games)) and all(
                len(game.game_states) == len(self.game_states) for game in GameHandler.games
            ):
                GameHandler.tqdm.update()
                game_state, previous_action = GameHandler.get_game_states_and_actions(
                    slice(-1, -2, -1)
                )
                if 'NN' in GameHandler.agents:
                    nn_actions = GameHandler.nn_agent(game_state, previous_action)
                if 'AI' in GameHandler.agents:
                    simple_actions = GameHandler.simple_agent(game_state, previous_action)

                i = 0
                for game in GameHandler.games:
                    game.game_actions.append([])
                    message = b''
                    for player in range(len(game.players)):
                        first_action = (
                            GameHandler.agents == 'AIvAI'
                            or (GameHandler.agents == 'AIvNN' and i % 2 == 0)
                            or (GameHandler.agents == 'NNvAI' and i % 2 == 1)
                        )
                        action = (simple_actions if first_action else nn_actions).apply(
                            lambda t: t[i]
                        )
                        masked_sidebar_action = GameAction(
                            action.button[~game_state.sidebar_mask[-1, i]],
                            action.mouse_x,
                            action.mouse_y,
                        )
                        game.game_actions[-1].append(masked_sidebar_action)
                        message += cnc_structs.ActionRequestArgs(
                            player,
                            masked_sidebar_action.button.nonzero()[0][0],
                            masked_sidebar_action.button.nonzero()[1][0],
                            masked_sidebar_action.mouse_x,
                            masked_sidebar_action.mouse_y,
                        ).render_message()
                        i += 1
                    game.write_message(message, binary=True)

    def open(self):
        GameHandler.games.append(self)
        self.set_nodelay(True)
        self.init_game()

    def on_close(self):
        GameHandler.ended_games.append(self)
        self.end_game()
        for game in GameHandler.games:
            if game not in GameHandler.ended_games:
                game.close()

        GameHandler.destroy_if_all_stopped()

    def add_players(self):
        colors = set(range(6))
        self.players = []
        for player in GameHandler.players:
            player = cnc_structs.CNCPlayerInfoStruct.from_buffer_copy(bytes(player))
            if player.ColorIndex < 0 or player.ColorIndex >= 6:
                player.ColorIndex = choice(list(colors))
                colors -= {player.ColorIndex}
            if player.House not in [0, 1]:
                player.House = choice([0, 1])

            self.write_message(player.render_message(), binary=True)
            self.players.append(player)

    def print_what_player_sees(self, player):
        game_state = self.game_states[-1][player]
        print(cnc_structs.render_game_state_terminal(game_state))

    def assess_players_performance(self):
        scores = self.compute_scores()
        loser_mask = sum(1 << i for i in range(len(self.players)) if scores[i] < max(scores))
        return loser_mask

    def compute_scores(self):
        scores = [0] * len(self.players)
        if len(self.game_states[-1]) > 0:
            for i, (player, game_state) in enumerate(zip(self.players, self.game_states[-1])):
                scores[i] = cnc_structs.score(game_state, player.ColorIndex)
        return scores

    def end_game(self):
        if self.loser_mask == 0:
            self.loser_mask = self.assess_players_performance()

    def init_game(self):
        self.folder = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S.%fs") + '_' + str(id(self))

        # communicate asset names
        buffer = bytes(ctypes.c_uint32(9))
        buffer += encode_list(cnc_structs.static_tile_names)
        self.write_message(buffer, binary=True)

        buffer = bytes(ctypes.c_uint32(10))
        buffer += encode_list(cnc_structs.dynamic_object_names)
        self.write_message(buffer, binary=True)

        # add players
        self.add_players()

        self.loser_mask = 0
        self.game_states = []
        # Actions are shifted by one
        # This is actually the previous action (None in the first turn)
        self.game_actions = [
            [GameAction(numpy.array([[True] + [False] * 11]), 744.0, 744.0) for _ in self.players]
        ]

        # start game
        buffer = self.start_game_args.render_message()
        self.write_message(buffer, binary=True)

    @classmethod
    def destroy_if_all_stopped(cls):
        cls.tqdm.update()
        if set(cls.games) == set(cls.ended_games):
            tornado.ioloop.IOLoop.current().stop()

    def save_gameplay(self):
        mkdir(self.folder)
        with open(self.folder + '/players.pkl', 'wb') as f:
            pickle.dump(self.players, f)
        with open(self.folder + '/loser_mask.txt', 'wt') as f:
            print(self.loser_mask, file=f)
        with open(self.folder + '/game_states.npy', 'wb') as f:
            numpy.save(f, self.game_states)
        with open(self.folder + '/game_actions.pkl', 'wb') as f:
            pickle.dump(self.game_actions, f)

    @classmethod
    def get_game_states_and_actions(cls, index: slice):
        max_length = min(len(game.game_states) for game in cls.games)
        n_players = sum(len(game.players) for game in cls.games)
        tensors = [
            player_state
            for time in range(max_length)[index]
            for game in cls.games
            for player_state in game.game_states[time]
        ]
        length = len(tensors) // n_players
        game_states = concatenate_game_states(tensors)
        time_aware_game_states = game_states.apply(
            lambda t: t.reshape(length, n_players, *t.shape[1:])
        )
        actions = concatenate_game_actions(
            [
                player_action
                for time in range(max_length)[index]
                for game in cls.games
                for player_action in game.game_actions[time]
            ]
        )
        time_aware_actions = actions.apply(lambda t: t.reshape(length, n_players, *t.shape[1:]))
        return time_aware_game_states, time_aware_actions

    @classmethod
    def split_per_games(cls, l):
        result = []
        l = iter(l)
        for game in cls.games:
            result.append([next(l) for _ in game.players])
        return result

    def get_rewards(self):
        rewards = [
            0 if self.loser_mask == 0 else (-1 if self.loser_mask & (1 << player) else 1)
            for player in range(len(self.players))
        ]
        if GameHandler.agents == 'AIvAI':
            rewards = numpy.ones_like(rewards)
        return rewards

    @classmethod
    def train(cls, n=1, time_window=200):
        game_state_tensor, actions = cls.get_game_states_and_actions(slice(None))
        rewards = numpy.concatenate([game.get_rewards() for game in cls.games])
        cls.nn_agent.learn(game_state_tensor, actions, rewards, n=n, time_window=time_window)

    @classmethod
    def configure(
        cls, nn_agent, players, start_game_args, n_games=2, end_limit=10_000, agents='NNvNN'
    ):
        cls.n_games = n_games
        cls.nn_agent = nn_agent
        cls.end_limit = end_limit
        cls.tqdm = trange(end_limit)
        cls.players = list(players)
        cls.start_game_args = start_game_args
        cls.games = []
        cls.ended_games = []
        cls.simple_agent = SimpleAgent()
        cls.agents = agents


def main():
    args = get_args()

    if args.load:
        agent = NNAgent.load(args.load)
    else:
        agent = NNAgent()

    GameHandler.configure(
        agent,
        [
            cnc_structs.CNCPlayerInfoStruct(
                GlyphxPlayerID=314159265,
                Name=b"ai1",
                House=127,
                Team=0,
                AllyFlags=0,
                ColorIndex=127,
                IsAI=False,
                StartLocationIndex=127,
            ),
            cnc_structs.CNCPlayerInfoStruct(
                GlyphxPlayerID=271828182,
                Name=b"ai2",
                House=127,
                Team=1,
                AllyFlags=0,
                ColorIndex=127,
                IsAI=False,
                StartLocationIndex=127,
            ),
        ],
        start_game_args=cnc_structs.StartGameArgs(
            cnc_structs.CNCMultiplayerOptionsStruct(
                MPlayerCount=2,
                MPlayerBases=1,
                MPlayerCredits=5000,
                MPlayerTiberium=1,
                MPlayerGoodies=0,
                MPlayerGhosts=0,
                MPlayerSolo=1,
                MPlayerUnitCount=0,
                IsMCVDeploy=False,
                SpawnVisceroids=False,
                EnableSuperweapons=True,
                MPlayerShadowRegrow=False,
                MPlayerAftermathUnits=True,
                CaptureTheFlag=False,
                DestroyStructures=False,
                ModernBalance=True,
            ),
            50,
            7,
            2,
        ),
        n_games=args.n,
        end_limit=args.end_limit,
        agents=args.agents,
    )

    agent.to(device=args.device, dtype=float16 if args.half else float32)

    application = tornado.web.Application([(r"/", GameHandler)])
    application.listen(args.port)
    if args.spawn:
        for i in range(args.n):
            tornado.ioloop.IOLoop.current().call_later(
                0.1 * i,
                lambda: Popen([args.exe, str(args.port), args.dir, args.dll]),
            )
    tornado.ioloop.IOLoop.current().start()
    GameHandler.tqdm.close()

    for game in GameHandler.games:
        print(game.folder, game.compute_scores(), game.get_rewards())
        if args.print:
            for i in range(len(game.players)):
                print(i)
                game.print_what_player_sees(i)
    if args.record:
        for game in GameHandler.games:
            game.save_gameplay()
    agent.to(device=args.device, dtype=float32)
    if args.train > 0:
        agent.init_optimizer(lr=args.learning_rate, weight_decay=1e-8)
        GameHandler.train(args.train, args.time_window)
    if args.save:
        agent.save(args.save)


if __name__ == '__main__':
    main()
