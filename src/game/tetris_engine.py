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
        ??????????????????????????????????????????????????????????????????????????????
        1??????????????????????????????
        2?????????????????????????????????????????????????????????
        3?????????board_state
        4????????????????????????????????????????????????board_state???????????????????????????????????????
        5??????????????????????????????????????????????????????????????????????????????????????????????????????
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

        # ????????????????????????????????????????????????
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

        # ??????????????????????????????????????????
        if np.any(cp_dst[0, :] > 0):
            game_stop_flag = True
            if clear_line_count > 0:
                print("#### ????????????????????????????????????????????????????????? ####")
                print("#### Before ####")
                print(res_board)
                print("#### After ####")
                print(cp_dst)
            return (
                game_stop_flag,
                clear_line_count,
                cp_dst[1:, :],
                cp_dst,
            )

        tmp_block = self.create_new_block()

        # ???????????????????????????????????????????????????
        tmp_game_state = tmp_block.draw(cp_dst[1:, :])

        # ?????????????????????????????????????????????????????????
        if np.any(tmp_game_state == Confs.init_value.value + Confs.solid_value.value):
            game_stop_flag = True
            # print("????????????????????????????????????????????????????????????")
            return (
                game_stop_flag,
                clear_line_count,
                cp_dst[1:, :],
                tmp_game_state,
            )

        self.board_state = np.copy(cp_dst[1:, :])
        self.tetromino_block = tmp_block

        return (
            game_stop_flag,
            clear_line_count,
            tmp_game_state,
            None,
        )

    def test_step(self, action: Action_Type):
        """
        ??????CNN?????????????????????????????????????????????????????????
        ???????????????????????????????????????????????????????????????????????????
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
        elif action == Action_Type.Bottom:
            # ??????????????????????????????????????????board_state?????????
            for step in range(Confs.row_count.value):
                is_dead, new_game_state = self.tetromino_block.move_down(
                    self.board_state
                )
                if is_dead:
                    (
                        is_end,
                        cleared_lines,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
                    return (
                        game_state,
                        cleared_lines * Confs.each_line_reward.value,
                        is_end,
                        debug,
                    )
            else:
                print("#### steps", step + 1)
                print("####", self.tetromino_block.type)
                print("####", self.board_state)
                raise Exception(f"implementation error for action [{action}]")
        elif action == Action_Type.Left_Down:
            # ?????????????????????????????????
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
                        cleared_lines,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
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
            else:
                return (
                    self.tetromino_block.draw(self.board_state),
                    Confs.operation_not_allowed_reward.value,
                    False,
                    imaginary_game_state,
                )
        elif action == Action_Type.Right_Down:
            # ?????????????????????????????????
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
                        cleared_lines,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
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
                        cleared_lines,
                        game_state,
                        debug,
                    ) = self.proc_when_block_dead(self.board_state)
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
            else:
                return (
                    self.tetromino_block.draw(self.board_state),
                    Confs.operation_not_allowed_reward.value,
                    False,
                    imaginary_game_state,
                )
        raise Exception(f"no implementation for action [{action}]")
