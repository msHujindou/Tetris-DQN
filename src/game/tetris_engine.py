import numpy as np
from game.confs import Action_Type, Confs
from game.confs import Block_Type
from game.tetromino import tetromino

for i in range(10):
    print(np.random.randint(0, 5))


class tetris_engine:
    def __init__(self, tetromino_type_list: list, action_type_list: list) -> None:
        self.tetromino_type_list = []
        self.action_type_list = []
        for item in tetromino_type_list:
            self.tetromino_type_list.append(item)
        for item in action_type_list:
            self.action_type_list.append(item)
        if len(tetromino_type_list) <= 0:
            self.tetromino_type_list.append(Block_Type.I)
            self.tetromino_type_list.append(Block_Type.O)
            self.tetromino_type_list.append(Block_Type.T)
            self.tetromino_type_list.append(Block_Type.Z)
            self.tetromino_type_list.append(Block_Type.S)
            self.tetromino_type_list.append(Block_Type.J)
            self.tetromino_type_list.append(Block_Type.L)
        if len(action_type_list) <= 0:
            self.action_type_list.append(Action_Type.Left)
            self.action_type_list.append(Action_Type.Right)
            self.action_type_list.append(Action_Type.Rotate)
            self.action_type_list.append(Action_Type.Down)
        self.action_space = len(self.action_type_list)

        self.board_state = np.zeros(
            (Confs.row_count.value, Confs.col_count.value), np.ubyte
        )
        # self.tetromino_block = tetromino(tetromino.)

    def reset(self):
        pass

    def step(self, action):
        new_state, reward, done, debug = None, None, None, None
        return new_state, reward, done, debug
