from enum import Enum


class Confs(Enum):
    row_count = 20
    col_count = 10
    init_value = 128
    blank_value = 0
    solid_value = 1
    operation_fail_reward = 0.0
    operation_succeed_but_cleared_zero_line_reward = 5.0
    reward_for_each_cleared_line = 200


class Block_Type(Enum):
    I = 0
    O = 1
    T = 2
    Z = 3
    S = 4
    L = 5
    J = 6


class Action_Type(Enum):
    Left = 0
    Right = 1
    Rotate = 2
    Down = 3
    Bottom = 4
    Left_Down = 5
    Right_Down = 6
    Rotate_Down = 7
