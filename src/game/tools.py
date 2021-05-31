import numpy as np
import confs
from game.tetromino import Block_Type


def get_calc_state(block_type: Block_Type, tmpx: int, tmpy: int, tmprotate: int):
    """
    为了方便计算，根据俄罗斯方块的类型，以及假想的坐标，生成一个假想的空的 ndarray，
    再将俄罗斯方块的信息填上去
    Args:
        block_type ([Block_Type]): [俄罗斯方块类型]
        tmpx ([int]): [假想的 X 坐标]
        tmpy ([int]): [假想的 Y 坐标]
        tmprotate ([int]): [假想的旋转度数]

    Returns:
        [np.ndarray]: [返回一个上面多出一行，下面多出两行，左右都多出两列的空ndarray]
    """
    # 新坐标
    tmpx, tmpy = tmpx + 2, tmpy + 1
    # 新矩阵
    virtual_state = np.zeros(
        (1 + confs.row_count + 2, 2 + confs.col_count + 2), np.ubyte
    )
    # 横条
    if block_type == Block_Type.I:
        if tmprotate == 0:
            virtual_state[tmpy, tmpx - 1 : tmpx + 3] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy - 1 : tmpy + 3, tmpx] = confs.init_value
    # 正方形
    if block_type == Block_Type.O:
        virtual_state[tmpy : tmpy + 2, tmpx : tmpx + 2] = confs.init_value

    # 山形
    if block_type == Block_Type.T:
        if tmprotate == 0:
            virtual_state[tmpy, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy - 1, tmpx] = confs.init_value
        if tmprotate == 180:
            virtual_state[tmpy, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy + 1, tmpx] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx] = confs.init_value
            virtual_state[tmpy, tmpx - 1] = confs.init_value
        if tmprotate == 270:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx] = confs.init_value
            virtual_state[tmpy, tmpx + 1] = confs.init_value

    # Z 形
    if block_type == Block_Type.Z:
        if tmprotate == 0:
            virtual_state[tmpy - 1, tmpx - 1 : tmpx + 1] = confs.init_value
            virtual_state[tmpy, tmpx : tmpx + 2] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy : tmpy + 2, tmpx - 1] = confs.init_value
            virtual_state[tmpy - 1 : tmpy + 1, tmpx] = confs.init_value

    # 反 Z 形
    if block_type == Block_Type.S:
        if tmprotate == 0:
            virtual_state[tmpy - 1, tmpx : tmpx + 2] = confs.init_value
            virtual_state[tmpy, tmpx - 1 : tmpx + 1] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy - 1 : tmpy + 1, tmpx - 1] = confs.init_value
            virtual_state[tmpy : tmpy + 2, tmpx] = confs.init_value

    # L 形
    if block_type == Block_Type.L:
        if tmprotate == 0:
            virtual_state[tmpy, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy - 1, tmpx - 1] = confs.init_value
        if tmprotate == 180:
            virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy, tmpx + 1] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx] = confs.init_value
            virtual_state[tmpy + 1, tmpx - 1] = confs.init_value
        if tmprotate == 270:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = confs.init_value
            virtual_state[tmpy - 1, tmpx] = confs.init_value

    # 反 L 形
    if block_type == Block_Type.J:
        if tmprotate == 0:
            virtual_state[tmpy, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy - 1, tmpx + 1] = confs.init_value
        if tmprotate == 180:
            virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = confs.init_value
            virtual_state[tmpy, tmpx - 1] = confs.init_value
        if tmprotate == 90:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx] = confs.init_value
            virtual_state[tmpy - 1, tmpx - 1] = confs.init_value
        if tmprotate == 270:
            virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = confs.init_value
            virtual_state[tmpy + 1, tmpx] = confs.init_value

    return virtual_state
