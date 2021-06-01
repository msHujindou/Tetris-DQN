import numpy as np
from game.confs import Confs
from game.confs import Block_Type
from game.tools import get_calc_state


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

    def draw(self, board_state: np.ndarray):
        pass

    def move_left(self, board_state: np.ndarray):
        is_movable, new_center_point, imaginary_board_state = self.can_move_left(
            board_state
        )
        if is_movable:
            self.center_point = new_center_point
            return imaginary_board_state
        else:
            return None

    def move_right(self, board_state: np.ndarray):
        pass

    def move_down(self, board_state: np.ndarray):
        calc_state = get_calc_state(
            self.type, self.center_point[1], self.center_point[0] + 1, self.rotate
        )
        # 移动到底部了，这个方块死掉，生成一个新的方块
        if np.any(calc_state[Confs.row_count.value + 1 :, :] == Confs.init_value.value):
            self.is_active = False
            # print("俄罗斯方块死亡 : 到达了底部后又继续向下移动")
            return True

        imaginary_board_state = (
            board_state
            + calc_state[1 : Confs.row_count.value + 1, 2 : Confs.col_count.value + 2]
        )
        # 往下移动的时候，被别的方块挡住了，这个方块死掉，生成一个新的方块
        if np.any(
            imaginary_board_state == Confs.init_value.value + Confs.solid_value.value
        ):
            self.is_active = False
            # print("俄罗斯方块死亡 : 被别的方块挡住又继续向下移动")
            return True

        self.center_point = (self.center_point[0] + 1, self.center_point[1])
        return False

    def move_bottom(self, board_state: np.ndarray):
        pass

    def rotate_on_board(self, board_state: np.ndarray):
        pass

    def can_move_left(self, board_state: np.ndarray):
        """
        判断当前的俄罗斯方块在game board上是否能够往左边移动
        1. 假如此方块已经在game board的左边，继续往左移动则会越界
        2. 假如此方块不在game board的左边，但左边有障碍物，继续移动会发生覆盖

        Args:
            board_state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        tmppoint = (self.center_point[0], self.center_point[1] - 1)
        calc_state = get_calc_state(
            self.type, self.center_point[1] - 1, self.center_point[0], self.rotate
        )
        # 判断左边是否越界
        if np.any(calc_state[:, 0:2] == Confs.init_value.value):
            # print("左边越界")
            return False, tmppoint, calc_state
        imaginary_board_state = (
            board_state
            + calc_state[1 : Confs.row_count.value + 1, 2 : Confs.col_count.value + 2]
        )
        # 判断左边是否有障碍物
        if np.any(
            imaginary_board_state == Confs.init_value.value + Confs.solid_value.value
        ):
            # print("左边有障碍物")
            return False, tmppoint, imaginary_board_state
        return True, tmppoint, imaginary_board_state

    def can_move_right(self, board_state: np.ndarray):
        """
        判断当前的俄罗斯方块在game board上是否能够往右边移动
        1. 假如此方块已经在game board的右边，继续往右移动则会越界
        2. 假如此方块不在game board的右边，但右边有障碍物，继续移动会发生覆盖

        Args:
            board_state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        tmppoint = (self.center_point[0], self.center_point[1] + 1)
        calc_state = get_calc_state(
            self.type, self.center_point[1] + 1, self.center_point[0], self.rotate
        )
        # 判断右边是否越界
        if np.any(calc_state[:, -2:-1] == Confs.init_value.value):
            # print("右边越界")
            return False, tmppoint, calc_state
        imaginary_board_state = (
            board_state
            + calc_state[1 : Confs.row_count.value + 1, 2 : Confs.col_count.value + 2]
        )
        # 判断右边是否有障碍物
        if np.any(
            imaginary_board_state == Confs.init_value.value + Confs.solid_value.value
        ):
            # print("右边有障碍物")
            return False, tmppoint, imaginary_board_state

        return True, tmppoint, imaginary_board_state

    def can_move_down(self, board_state: np.ndarray):
        pass

    def can_move_bottom(self, board_state: np.ndarray):
        pass

    def can_rotate_on_board(self, board_state: np.ndarray):
        """
        判断当前的俄罗斯方块在game board上是否能旋转
        1. 假如此方块是正方块，则返回False
        2. 假如此方块旋转后，左边越界、右边越界、底部越界，则返回False
        3. 假如此方块旋转后，和别的重合，则返回False

        Args:
            board_state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
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
            return False, tmprotate, None
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
        if np.any(calc_state[:, 0:2] == Confs.init_value.value):
            # print("旋转失败，左边越界")
            return False, tmprotate, calc_state
        if np.any(calc_state[:, -2:-1] == Confs.init_value.value):
            # print("旋转失败，右边越界")
            return False, tmprotate, calc_state
        if np.any(calc_state[Confs.row_count.value + 1 :, :] == Confs.init_value.value):
            # print("旋转失败，底部越界")
            return False, tmprotate, calc_state
        imaginary_board_state = (
            board_state
            + calc_state[1 : Confs.row_count.value + 1, 2 : Confs.col_count.value + 2]
        )
        if np.any(
            imaginary_board_state == Confs.init_value.value + Confs.solid_value.value
        ):
            # print("旋转失败，和别的方块重合")
            return False, tmprotate, imaginary_board_state

        return True, tmprotate, imaginary_board_state
