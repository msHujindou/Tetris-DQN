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
        self.max_random_index = len(self.tetromino_type_list)
        self.board_state = None
        self.tetromino_block = None

    def select_random_step(self):
        action_index = int(np.random.randint(0, self.action_space))
        return action_index, self.action_type_list[action_index]

    def select_random_step_action(self, action_index):
        return self.action_type_list[action_index]

    def reset(self):
        self.board_state = np.zeros(
            (Confs.row_count.value + 1, Confs.col_count.value), np.ubyte
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

        # res_board = virtual_state
        res_board = board_state + virtual_state

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
            if clear_line_count > 0:
                print("#### 消除可以消除的行之后，仍然有越界的方块 ####")
                print("#### Before ####")
                print(res_board)
                print("#### After ####")
                print(cp_dst)
            return (
                game_stop_flag,
                clear_line_count * Confs.each_line_reward.value,
                cp_dst,
                cp_dst,
            )

        tmp_block = self.create_new_block()

        # 检查此随机生成的方块能否放在局面上
        tmp_game_state = tmp_block.draw(cp_dst)

        # 检查新生成的方块是否和死掉的方块有重合
        if np.any(tmp_game_state == Confs.init_value.value + Confs.solid_value.value):
            game_stop_flag = True
            # print("游戏结束，因为没有空间容纳新的俄罗斯方块")
            return (
                game_stop_flag,
                clear_line_count * Confs.each_line_reward.value,
                cp_dst,
                tmp_game_state,
            )

        self.board_state = np.copy(cp_dst)
        self.tetromino_block = tmp_block

        return (
            game_stop_flag,
            clear_line_count * Confs.each_line_reward.value,
            tmp_game_state,
            None,
        )

    def calc_agg_height(self, board_state: np.ndarray):
        res = 0
        for col in range(board_state.shape[1]):
            idx = np.argmax(board_state[:, col] == Confs.solid_value.value)
            if idx != 0:
                idx = board_state.shape[0] - idx
            res += idx
            # print("agg_height", idx)
        return res

    def calc_holes(self, board_state: np.ndarray):
        res = 0
        for col in range(board_state.shape[1]):
            for row in range(1, board_state.shape[0]):
                if board_state[row, col] == Confs.blank_value.value:
                    if row == 1:
                        if (
                            col > 0
                            and col < board_state.shape[1] - 1
                            and np.any(
                                board_state[row, col + 1 :] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row, 0:col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row + 1 :, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                        elif (
                            col == 0
                            and np.any(
                                board_state[row, col + 1 :] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row + 1 :, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                        elif (
                            col == board_state.shape[1] - 1
                            and np.any(
                                board_state[row, 0:col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row + 1 :, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                    elif row == board_state.shape[0] - 1:
                        if (
                            col > 0
                            and col < board_state.shape[1] - 1
                            and np.any(
                                board_state[row, col + 1 :] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row, 0:col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[1:row, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                        elif (
                            col == 0
                            and np.any(
                                board_state[row, col + 1 :] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[1:row, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                        elif (
                            col == board_state.shape[1] - 1
                            and np.any(
                                board_state[row, 0:col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[1:row, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                    elif col == 0:
                        if (
                            row > 1
                            and row < board_state.shape[0] - 1
                            and np.any(
                                board_state[row, col + 1 :] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[1:row, col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row + 1 :, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                    elif col == board_state.shape[1] - 1:
                        if (
                            row > 1
                            and row < board_state.shape[0] - 1
                            and np.any(
                                board_state[row, 0:col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[1:row, col] == Confs.solid_value.value
                            )
                            and np.any(
                                board_state[row + 1 :, col] == Confs.solid_value.value
                            )
                        ):
                            # print("Hole Found", row, col)
                            res += 1
                    elif (
                        np.any(board_state[row, col + 1 :] == Confs.solid_value.value)
                        and np.any(board_state[row, 0:col] == Confs.solid_value.value)
                        and np.any(board_state[1:row, col] == Confs.solid_value.value)
                        and np.any(
                            board_state[row + 1 :, col] == Confs.solid_value.value
                        )
                    ):
                        # print("Hole Found", row, col)
                        res += 1

        return res

    def calc_bumpiness(self, board_state: np.ndarray):
        res = 0
        for col in range(board_state.shape[1]):
            col_next = col + 1
            if col_next < board_state.shape[1]:
                idx1 = np.argmax(board_state[:, col] == Confs.solid_value.value)
                if idx1 != 0:
                    idx1 = board_state.shape[0] - idx1
                idx2 = np.argmax(board_state[:, col_next] == Confs.solid_value.value)
                if idx2 != 0:
                    idx2 = board_state.shape[0] - idx2
                res += np.absolute(idx1 - idx2)
                # print("bumpiness", np.absolute(idx1 - idx2))
        return res

    def proc_when_block_dead2(self, board_state: np.ndarray):
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

        # res_board = virtual_state
        res_board = board_state + virtual_state

        cp_dst = np.zeros((Confs.row_count.value + 1, Confs.col_count.value), np.ubyte)
        cp_idx = 0

        clear_line_count = 0
        game_stop_flag = False

        # https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
        agg_height = self.calc_agg_height(res_board)
        holes = self.calc_holes(res_board)
        bumpiness = self.calc_bumpiness(res_board)

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

        rwd = (
            -0.510066 * agg_height
            + 10.0 * clear_line_count
            - 0.35663 * holes
            - 0.184483 * bumpiness
        )
        # print(
        #     "agg_height, clear_line_count, holes, bumpiness",
        #     agg_height,
        #     clear_line_count,
        #     holes,
        #     bumpiness,
        # )

        # 先消除行、再检查是否有越界的
        if np.any(cp_dst[0, :] > 0):
            game_stop_flag = True
            if clear_line_count > 0:
                print("#### 消除可以消除的行之后，仍然有越界的方块 ####")
                print("#### Before ####")
                print(res_board)
                print("#### After ####")
                print(cp_dst)
            return (
                game_stop_flag,
                rwd,
                cp_dst,
                cp_dst,
            )

        tmp_block = self.create_new_block()

        # 检查此随机生成的方块能否放在局面上
        tmp_game_state = tmp_block.draw(cp_dst)

        # 检查新生成的方块是否和死掉的方块有重合
        if np.any(tmp_game_state == Confs.init_value.value + Confs.solid_value.value):
            game_stop_flag = True
            # print("游戏结束，因为没有空间容纳新的俄罗斯方块")
            return (
                game_stop_flag,
                rwd,
                cp_dst,
                tmp_game_state,
            )

        self.board_state = np.copy(cp_dst)
        self.tetromino_block = tmp_block

        return (
            game_stop_flag,
            rwd,
            tmp_game_state,
            None,
        )

    def test_step(self, action: Action_Type):
        """
        检测CNN是否能识别出活着的俄罗斯方块在任意盘局
        可以往左走的步数、可以往右走的步数、可以旋转的次数
        Args:
            action (Action_Type): [description]
        """
        if action == Action_Type.Left:
            center_point = self.tetromino_block.center_point
            movable_step = 0
            while True:
                (
                    is_movable,
                    new_center_point,
                    imaginary_game_state,
                ) = self.tetromino_block.can_move_left(self.board_state, center_point)
                if is_movable:
                    movable_step += 1
                    center_point = new_center_point
                else:
                    break
            return movable_step
        elif action == Action_Type.Right:
            center_point = self.tetromino_block.center_point
            movable_step = 0
            while True:
                (
                    is_movable,
                    new_center_point,
                    imaginary_game_state,
                ) = self.tetromino_block.can_move_right(self.board_state, center_point)
                if is_movable:
                    movable_step += 1
                    center_point = new_center_point
                else:
                    break
            return movable_step
        elif action == Action_Type.Down:
            center_point = self.tetromino_block.center_point
            movable_step = 0
            while True:
                (
                    is_movable,
                    will_dead_if_force_move,
                    new_center_point,
                    imaginary_game_state,
                ) = self.tetromino_block.can_move_down(self.board_state, center_point)
                if is_movable:
                    movable_step += 1
                    center_point = new_center_point
                else:
                    break
            return movable_step
        elif action == Action_Type.Rotate:
            if self.tetromino_block.type == Block_Type.O:
                return 0
            current_rotate = self.tetromino_block.rotate
            rotate_step = 0
            while True:
                (
                    is_rotatable,
                    new_rotate,
                    imaginary_game_state,
                ) = self.tetromino_block.can_rotate_on_board(
                    self.board_state, current_rotate
                )
                if is_rotatable:
                    rotate_step += 1
                    current_rotate = new_rotate
                    if rotate_step >= 4:
                        break
                else:
                    break
            if (
                self.tetromino_block.type == Block_Type.I
                or self.tetromino_block.type == Block_Type.Z
                or self.tetromino_block.type == Block_Type.S
            ):
                if rotate_step >= 2:
                    return 2
                else:
                    return rotate_step
            if (
                self.tetromino_block.type == Block_Type.T
                or self.tetromino_block.type == Block_Type.L
                or self.tetromino_block.type == Block_Type.J
            ):
                if rotate_step >= 4:
                    return 4
                else:
                    return rotate_step

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
                is_end, reward, game_state, debug = self.proc_when_block_dead(
                    self.board_state
                )
                return (
                    game_state,
                    reward,
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
        elif action == Action_Type.Bottom:
            # 能向下移动的最大步数不会超过board_state的行数
            for step in range(Confs.row_count.value):
                is_dead, new_game_state = self.tetromino_block.move_down(
                    self.board_state
                )
                if is_dead:
                    (
                        is_end,
                        reward,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
                    return (
                        game_state,
                        reward,
                        is_end,
                        debug,
                    )
            else:
                print("#### steps", step + 1)
                print("####", self.tetromino_block.type)
                print("####", self.board_state)
                raise Exception(f"implementation error for action [{action}]")
        elif action == Action_Type.Left_Down:
            # 向左移动后，并向下移动
            (
                is_movable,
                new_center_point,
                imaginary_game_state,
            ) = self.tetromino_block.can_move_left(self.board_state)
            if is_movable:
                self.tetromino_block.center_point = new_center_point

                is_dead, new_game_state = self.tetromino_block.move_down(
                    self.board_state
                )
                if is_dead:
                    (
                        is_end,
                        reward,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
                    return (
                        game_state,
                        reward,
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
            else:
                return (
                    self.tetromino_block.draw(self.board_state),
                    Confs.operation_not_allowed_reward.value,
                    False,
                    imaginary_game_state,
                )
        elif action == Action_Type.Right_Down:
            # 向右移动后，并向下移动
            (
                is_movable,
                new_center_point,
                imaginary_game_state,
            ) = self.tetromino_block.can_move_right(self.board_state)
            if is_movable:
                self.tetromino_block.center_point = new_center_point

                is_dead, new_game_state = self.tetromino_block.move_down(
                    self.board_state
                )
                if is_dead:
                    (
                        is_end,
                        reward,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
                    return (
                        game_state,
                        reward,
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
            else:
                return (
                    self.tetromino_block.draw(self.board_state),
                    Confs.operation_not_allowed_reward.value,
                    False,
                    imaginary_game_state,
                )
        elif action == Action_Type.Rotate_Down:
            (
                can_rotate,
                new_rotate,
                imaginary_game_state,
            ) = self.tetromino_block.can_rotate_on_board(self.board_state)
            if can_rotate:
                self.tetromino_block.rotate = new_rotate
                is_dead, new_game_state = self.tetromino_block.move_down(
                    self.board_state
                )
                if is_dead:
                    (
                        is_end,
                        reward,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead2(self.board_state)
                    return (
                        game_state,
                        reward,
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
            else:
                return (
                    self.tetromino_block.draw(self.board_state),
                    Confs.operation_not_allowed_reward.value,
                    False,
                    imaginary_game_state,
                )
        raise Exception(f"no implementation for action [{action}]")
