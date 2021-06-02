import numpy as np
from game.confs import Action_Type, Confs
from game.confs import Block_Type
from game.tetromino import tetromino
from game.tools import get_calc_state


class tetris_engine:
    def __init__(
        self, tetromino_type_list: list = None, action_type_list: list = None
    ) -> None:
        self.tetromino_type_list = []
        self.action_type_list = []
        if tetromino_type_list:
            for item in tetromino_type_list:
                self.tetromino_type_list.append(item)
        if action_type_list:
            for item in action_type_list:
                self.action_type_list.append(item)
        if len(self.tetromino_type_list) <= 0:
            self.tetromino_type_list.append(Block_Type.I)
            self.tetromino_type_list.append(Block_Type.O)
            self.tetromino_type_list.append(Block_Type.T)
            self.tetromino_type_list.append(Block_Type.Z)
            self.tetromino_type_list.append(Block_Type.S)
            self.tetromino_type_list.append(Block_Type.J)
            self.tetromino_type_list.append(Block_Type.L)
        if len(self.action_type_list) <= 0:
            self.action_type_list.append(Action_Type.Left)
            self.action_type_list.append(Action_Type.Right)
            self.action_type_list.append(Action_Type.Rotate)
            self.action_type_list.append(Action_Type.Down)
        self.action_space = len(self.action_type_list)

        self.board_state = np.zeros(
            (Confs.row_count.value, Confs.col_count.value), np.ubyte
        )
        self.max_random_index = len(self.tetromino_type_list)
        self.tetromino_block = tetromino(
            self.tetromino_type_list[int(np.random.randint(0, self.max_random_index))]
        )

    def reset(self):
        self.board_state = np.zeros(
            (Confs.row_count.value, Confs.col_count.value), np.ubyte
        )
        self.tetromino_block = tetromino(
            self.tetromino_type_list[int(np.random.randint(0, self.max_random_index))]
        )
        return self.tetromino_block.draw(self.board_state)

    def create_new_block(self):
        return tetromino(
            self.tetromino_type_list[int(np.random.randint(0, self.max_random_index))]
        )

    def proc_when_block_dead(self, board_state: np.ndarray):
        """
        当一个俄罗斯方块死亡后的处理函数，此函数的功能如下：
        1、计算可以消除的行数
        2、判断消除完行后是否存在顶部越界的情况
        3、更新board_state
        4、生成新的俄罗斯方块，并判断新的board_state是否能容纳下新的俄罗斯方块
        5、无论是顶部越界或者是无法容纳新生成的俄罗斯方块，都返回游戏结束标志
        Args:
            board_state ([type]): [description]

        Returns:
            [type]: [description]
        """
        virtual_state = get_calc_state(
            self.tetromino_block.type,
            self.tetromino_block.center_point[1],
            self.tetromino_block.center_point[0],
            self.tetromino_block.rotate,
            Confs.solid_value.value,
        )

        virtual_state = virtual_state[
            0 : Confs.row_count.value + 1, 2 : Confs.col_count.value + 2
        ]

        res_board = virtual_state
        res_board[1:, :] = board_state + virtual_state[1:, :]

        cp_dst = np.zeros((Confs.row_count.value + 1, Confs.col_count.value), np.ubyte)
        cp_idx = 0

        clear_line_count = 0
        game_stop_flag = False

        # 逐行检查、查看是否有可以消去的行
        for i in range(res_board.shape[0]):
            if np.all(
                res_board[res_board.shape[0] - i - 1, :] == Confs.solid_value.value
            ):
                clear_line_count += 1
            else:
                cp_dst[res_board.shape[0] - cp_idx - 1, :] = res_board[
                    res_board.shape[0] - i - 1, :
                ]
                cp_idx += 1

        # 先消除行、再检查是否有越界的
        if np.any(cp_dst[0, :] > 0):
            game_stop_flag = True
            print("#### 消除可以消除的行之后，仍然有越界的方块 ####")
            print("#### Before ####")
            print(res_board)
            print("#### After ####")
            print(cp_dst)
            return (
                game_stop_flag,
                clear_line_count,
                cp_dst,
                res_board,
            )

        tmp_block = self.create_new_block()

        # 检查此随机生成的方块能否放在局面上
        tmp_game_state = tmp_block.draw(cp_dst[1:, :])

        # 检查新生成的方块是否和死掉的方块有重合
        if np.any(tmp_game_state == Confs.init_value.value + Confs.solid_value.value):
            game_stop_flag = True
            # print("游戏结束，因为没有空间容纳新的俄罗斯方块")
            return (
                game_stop_flag,
                clear_line_count,
                tmp_game_state,
                None,
            )

        self.board_state = np.copy(cp_dst[1:, :])
        self.tetromino_block = tmp_block

        return (
            game_stop_flag,
            clear_line_count,
            tmp_game_state,
            None,
        )

    def step(self, action: Action_Type):
        """[summary]

        Args:
            action (Action_Type): [description]

        Raises:
            Exception: [description]

        Returns:
            [type]: [ new_state , reward , is_end , debug_info ]
        """
        if action == Action_Type.Left:
            return self.tetromino_block.move_left(self.board_state)
        elif action == Action_Type.Right:
            return self.tetromino_block.move_right(self.board_state)
        elif action == Action_Type.Rotate:
            return self.tetromino_block.rotate_on_board(self.board_state)
        elif action == Action_Type.Down:
            is_dead, new_game_state = self.tetromino_block.move_down(self.board_state)
            if is_dead:
                is_end, cleared_lines, game_state, debug = self.proc_when_block_dead(
                    self.board_state
                )
                return (
                    game_state,
                    cleared_lines * Confs.each_line_reward.value,
                    is_end,
                    debug,
                )
            else:
                return (
                    new_game_state,
                    Confs.move_down_no_line_cleared_reward.value,
                    False,
                    None,
                )
        raise Exception(f"no implementation for action [{action}]")
