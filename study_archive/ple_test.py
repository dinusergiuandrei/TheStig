import numpy as np
import os
from gym.envs.box2d.car_racing import CarRacing


os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"


from ple.games.pong import Pong
from ple import PLE

env = Pong()
game = env

p = PLE(game, fps=30, display_screen=True, force_fps=False)
p.init()

# myAgent = MyAgent(p.getActionSet())
# nb_frames = 1000
# reward = 0.0
#
# for f in range(nb_frames):
#     if p.game_over():  # check if the game is over
#         p.reset_game()
#
#     obs = p.getScreenRGB()
#     action = myAgent.pickAction(reward, obs)
#     reward = p.act(action)
