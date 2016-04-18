import numpy as np

class SimpleBreakoutVisualizer:
    def __init__(self, algo):
        import pygame
        self.screen_size = 12
        self.screen = pygame.display.set_mode((self.screen_size * 10, self.screen_size * 10 * 4))
        self.mem = {}

        self.algo = algo

    def show(self, prev_frames):
        import pygame

        def l(x):
            if x not in self.mem:
                self.mem[x] = (x, x, x)
            return self.mem[x]

        f_l = np.frompyfunc(l, 1, 3)
        rect = pygame.Surface((self.screen_size * 10, self.screen_size * 10 * 4))
        image = np.reshape(zip(*list(f_l(np.concatenate(prev_frames).flatten()))), (self.screen_size * 4, self.screen_size, 3))

        image = np.transpose(image, [1, 0, 2])
        pygame.surfarray.blit_array(rect, np.repeat(np.repeat(image, 10, axis=0), 10, axis=1))
        self.screen.blit(rect, (0, 0))

        pygame.display.flip()
        import time
        time.sleep(0.01)

    def next_game(self):
        pass


class SimpleBreakout(object):

    def __init__(self):
        self.action_set = [4, 7, 10]
        self.reset_game()
        import random
        self.bar = [random.randint(0, 11), 10]
        self.h = 12
        self.w = 12

        self.prev_frames = [np.zeros((self.h, self.w), dtype=np.uint8),
                            np.zeros((self.h, self.w), dtype=np.uint8),
                            np.zeros((self.h, self.w), dtype=np.uint8),
                            np.zeros((self.h, self.w), dtype=np.uint8)]

    def reset_game(self):
        import random
        self.finished = False
        self.ball = [random.randint(0, 11), 0]
        self.cum_reward = 0

    def n_actions(self):
        return len(self.action_set)

    def input(self, action):
        if action == 0:
            self.bar[0] = max(self.bar[0] - 1, 0)
        if action == 1:
            self.bar[0] = min(self.bar[0] + 1, 11)

        self.ball[1] += 1
        # print(self.bar, action)
        action_reward = 0
        game_over = False
        lol = False
        if self.ball[1] == 11:
            if abs(self.ball[0] - self.bar[0]) <= 1:
                action_reward = 1
                self.reset_game()
            else:
                game_over = True
                lol = True
                self.reset_game()
                import random
                self.bar = [random.randint(0,11), 11]

        self.cum_reward += action_reward
        self.prev_frames = self.prev_frames[1:]
        self.prev_frames.append(self._as_frame())
        self.finished = game_over

        return action_reward, lol

    def _as_frame(self):
        frame = np.zeros((12, 12), dtype=np.uint8)
        frame[self.ball[0], max(0, self.ball[1] - 1)] = 125
        frame[self.ball[0], min(11, self.ball[1] + 1)] = 125
        frame[self.ball[0], self.ball[1]] = 230
        frame[self.bar[0], self.bar[1]] = 200
        frame[max(0, self.bar[0]-1), self.bar[1]] = 180
        frame[min(11, self.bar[0]+1), self.bar[1]] = 180
        return frame

    def get_state(self):
        return self.prev_frames
