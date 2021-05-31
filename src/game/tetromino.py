from typing import Tuple
import numpy as np
from enum import Enum
import confs
from game.tools import get_calc_state


class Block_Type(Enum):
    I = 0
    O = 1
    T = 2
    Z = 3
    S = 4
    L = 5
    J = 6


class tetromino:
    def __init__(self, tetromino_type: Block_Type) -> None:
        self.type = tetromino_type
        self.rotate = 0
        # 活性则显示为红色、非活性显示为黑色
        self.is_active = True
        # I 形的中点
        if self.type == Block_Type.I:
            self.center_point = (0, 4)
        # O 形的中点
        if self.type == Block_Type.O:
            self.center_point = (-1, 4)
        # T 形的中点
        if self.type == Block_Type.T:
            self.center_point = (0, 5)
        # Z 形的中点
        if self.type == Block_Type.Z:
            self.center_point = (0, 5)
        # S 形的中点
        if self.type == Block_Type.S:
            self.center_point = (0, 5)
        # L 形的中点
        if self.type == Block_Type.L:
            self.center_point = (0, 5)
        # J 形的中点
        if self.type == Block_Type.J:
            self.center_point = (0, 5)

    def draw_on_board(self, board_state: np.ndarray):
        pass

    def move_left_on_board(self, board_state: np.ndarray):
        tmpstate = get_calc_state(
            self.type, self.center_point[1] - 1, self.center_point[0], self.rotate
        )
        if np.any(tmpstate[:, 0:2] == confs.init_value):
            # print("已经到达左边边界，无法继续往左移动")
            return False
        tmpboard = (
            board_state + tmpstate[1 : confs.row_count + 1, 2 : confs.col_count + 2]
        )
        if np.any(tmpboard == confs.init_value + confs.solid_value):
            # print("左边被方块挡住了，没法往左移动")
            return False

        self.center_point = (self.center_point[0], self.center_point[1] - 1)
        return True

    def move_right_on_board(self, board_state: np.ndarray):
        pass

    def move_down_on_board(self, board_state: np.ndarray):
        pass

    def move_bottom_on_board(self, board_state: np.ndarray):
        pass

    def rotate_on_board(self, board_state: np.ndarray):
        pass

    def can_move_left_on_board(self, board_state: np.ndarray):
        """
        判断当前的俄罗斯方块在game board上是否能够往左边移动
        1. 假如此方块已经在game board的左边，继续往左移动则会越界
        2. 加入此方块不在game board的左边，但左边有障碍物，继续移动会发生覆盖

        Args:
            board_state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        calc_state = get_calc_state(
            self.type, self.center_point[1] - 1, self.center_point[0], self.rotate
        )
        # 判断左边是否越界
        if np.any(calc_state[:, 0:2] == confs.init_value):
            # print("左边越界")
            return False, calc_state
        imaginary_board_state = (
            board_state + calc_state[1 : confs.row_count + 1, 2 : confs.col_count + 2]
        )
        # 判断左边是否有障碍物
        if np.any(imaginary_board_state == confs.init_value + confs.solid_value):
            # print("左边有障碍物")
            return False, imaginary_board_state
        return True, imaginary_board_state

    def can_move_right_on_board(self, board_state: np.ndarray):
        """
        判断当前的俄罗斯方块在game board上是否能够往左边移动
        1. 假如此方块已经在game board的左边，继续往左移动则会越界
        2. 加入此方块不在game board的左边，但左边有障碍物，继续移动会发生覆盖

        Args:
            board_state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        calc_state = get_calc_state(
            self.type, self.center_point[1] + 1, self.center_point[0], self.rotate
        )
        # 判断右边是否越界
        if np.any(calc_state[:, -2:-1] == confs.init_value):
            # print("右边越界")
            return False, calc_state
        imaginary_board_state = (
            board_state + calc_state[1 : confs.row_count + 1, 2 : confs.col_count + 2]
        )
        # 判断右边是否有障碍物
        if np.any(imaginary_board_state == confs.init_value + confs.solid_value):
            # print("右边有障碍物")
            return False, imaginary_board_state

        self.center_point = (self.center_point[0], self.center_point[1] + 1)
        return True, imaginary_board_state

    def can_move_down_on_board(self, board_state: np.ndarray):
        pass

    def can_move_bottom_on_board(self, board_state: np.ndarray):
        pass

    def can_rotate_on_board(self, board_state: np.ndarray):
        tmprotate = self.rotate
        # 一字形、z形、反z形的旋转
        if (
            self.type == Block_Type.I
            or self.type == Block_Type.Z
            or self.type == Block_Type.S
        ):
            if tmprotate == 0:
                tmprotate = 90
            else:
                tmprotate = 0
        # 正方块的旋转，总是返回False
        if self.type == Block_Type.O:
            tmprotate = 0
            return False, tmprotate
        # 山形、L形、反L形的旋转
        if (
            self.type == Block_Type.T
            or self.type == Block_Type.L
            or self.type == Block_Type.J
        ):
            # 旋转 90、180、270、0、90、180、270、0 ...
            if tmprotate < 270:
                tmprotate += 90
            else:
                tmprotate = 0

        calc_state = get_calc_state(
            self.type,
            self.center_point[1],
            self.center_point[0],
            tmprotate,
        )
        if np.any(calc_state[:, 0:2] == confs.init_value):
            # print("旋转失败，左边越界")
            return False, tmprotate
        if np.any(calc_state[:, -2:-1] == confs.init_value):
            # print("旋转失败，右边越界")
            return False, tmprotate
        if np.any(calc_state[confs.row_count + 1 :, :] == confs.init_value):
            # print("旋转失败，底部越界")
            return False, tmprotate
        imaginary_board_state = (
            board_state + calc_state[1 : confs.row_count + 1, 2 : confs.col_count + 2]
        )
        if np.any(imaginary_board_state == confs.init_value + confs.solid_value):
            # print("旋转失败，和别的方块重合")
            return False, tmprotate

        return True, tmprotate
