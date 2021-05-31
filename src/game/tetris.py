"""
此脚本负责 - 定义俄罗斯方块
board_state : 俄罗斯方块基本盘（除去红色俄罗斯方块后的背景）的状态
game_state : 由每一个活着的俄罗斯方块 + board_state之后的状态
"""
import numpy as np

row_count, col_count = 20, 10

init_value = 128
blank_value = 0
solid_value = 1

operation_fail_reward = 0.0
operation_succeed_but_cleared_zero_line_reward = 5.0
reward_for_each_cleared_line = 200


class block:
    def __init__(self):

        self.block_type = np.random.randint(1, 8)

        self.center_point = (0, 0)
        self.rotate = 0
        # 活性则显示为红色、非活性显示为黑色
        self.is_active = True
        # 一字形的中点
        if self.block_type == 1:
            self.center_point = (0, 4)
        # 正方块的中点
        if self.block_type == 2:
            self.center_point = (-1, 4)
        # 山形的中点
        if self.block_type == 3:
            self.center_point = (0, 5)
        # Z 形的中点
        if self.block_type == 4:
            self.center_point = (0, 5)
        # 反 Z 形的中点
        if self.block_type == 5:
            self.center_point = (0, 5)
        # L 形的中点
        if self.block_type == 6:
            self.center_point = (0, 5)
        # 反 L 形的中点
        if self.block_type == 7:
            self.center_point = (0, 5)

    def step(self, act, board_state):
        """
        # 0 -> 左移动
        # 1 -> 右移动
        # 2 -> 旋转
        # 3 -> 往下移动
        # 4 -> 往下移动到底部
        返回
        1. 游戏是否结束
        2. 消除的行数 (0-4)
        3. 新的俄罗斯方块
           如果当前方块尚未死掉，返回空
        4. 最终的局面
           消除后行的局面、或者加入新方块的局面、结束的话返回空
        5. debug用的局面
        6. 背景局面(用来计算局面)
        """

        game_stop_flag = False
        reward = 0
        tmp_block = None

        # 向左移动
        if act == 0:
            if self.move_left(board_state):
                #####################################
                # to speed up training , force move down after move left
                if self.move_down(board_state):
                    # 更新背景
                    return self.proc_when_block_dead(board_state)
                else:
                    game_state = self.get_state(board_state)
                    return (
                        game_stop_flag,
                        operation_succeed_but_cleared_zero_line_reward,
                        tmp_block,
                        game_state,
                        None,
                        board_state,
                    )
                #####################################
                game_state = self.get_state(board_state)
                return (
                    game_stop_flag,
                    reward,
                    tmp_block,
                    game_state,
                    None,
                    board_state,
                )
            else:
                # 操作不成功，返回原来的局面
                return (
                    game_stop_flag,
                    operation_fail_reward,
                    tmp_block,
                    None,
                    None,
                    board_state,
                )
        # 向右移动
        elif act == 1:
            if self.move_right(board_state):
                #####################################
                # to speed up training , force move down after move right
                if self.move_down(board_state):
                    # 更新背景
                    return self.proc_when_block_dead(board_state)
                else:
                    game_state = self.get_state(board_state)
                    return (
                        game_stop_flag,
                        operation_succeed_but_cleared_zero_line_reward,
                        tmp_block,
                        game_state,
                        None,
                        board_state,
                    )
                #####################################
                game_state = self.get_state(board_state)
                return (
                    game_stop_flag,
                    reward,
                    tmp_block,
                    game_state,
                    None,
                    board_state,
                )
            else:
                # 操作不成功，返回原来的局面
                return (
                    game_stop_flag,
                    operation_fail_reward,
                    tmp_block,
                    None,
                    None,
                    board_state,
                )
        # 旋转
        elif act == 2:
            if self.rotate_block(board_state):
                #####################################
                # to speed up training , force move down after rotate
                if self.move_down(board_state):
                    # 更新背景
                    return self.proc_when_block_dead(board_state)
                else:
                    game_state = self.get_state(board_state)
                    return (
                        game_stop_flag,
                        operation_succeed_but_cleared_zero_line_reward,
                        tmp_block,
                        game_state,
                        None,
                        board_state,
                    )
                #####################################
                game_state = self.get_state(board_state)
                return (
                    game_stop_flag,
                    reward,
                    tmp_block,
                    game_state,
                    None,
                    board_state,
                )
            else:
                # 操作不成功，返回原来的局面
                return (
                    game_stop_flag,
                    operation_fail_reward,
                    tmp_block,
                    None,
                    None,
                    board_state,
                )
        # 向下移动
        elif act == 3:
            if self.move_down(board_state):
                # 更新背景
                return self.proc_when_block_dead(board_state)
            else:
                game_state = self.get_state(board_state)
                return (
                    game_stop_flag,
                    operation_succeed_but_cleared_zero_line_reward,
                    tmp_block,
                    game_state,
                    None,
                    board_state,
                )
        elif act == 4:
            # 直接掉落到底部
            if self.move_bottom(board_state):
                # 更新背景
                return self.proc_when_block_dead(board_state)
            else:
                raise Exception("Move to bottom failed !!!")

        return game_stop_flag, reward, tmp_block, None, None, board_state

    def preview_init(self, board_state: np.ndarray):
        """
        在现有局面的上，假设生成一个新方块
        返回假想的局面和方块界面
        """
        virtual_state = np.zeros((row_count + 1, col_count), np.ubyte)
        value_to_set = init_value
        tmpx, tmpy = self.center_point[1], self.center_point[0] + 1
        # 横条
        if self.block_type == 1:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 3] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 3, tmpx] = value_to_set
        # 正方形
        if self.block_type == 2:
            virtual_state[tmpy : tmpy + 2, tmpx : tmpx + 2] = value_to_set
        # 山形
        if self.block_type == 3:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set
        # Z 形
        if self.block_type == 4:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 1] = value_to_set
                virtual_state[tmpy, tmpx : tmpx + 2] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1 : tmpy + 1, tmpx] = value_to_set
        # 反 Z 形
        if self.block_type == 5:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1 : tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 1, tmpx - 1] = value_to_set
                virtual_state[tmpy : tmpy + 2, tmpx] = value_to_set
        # L 形
        if self.block_type == 6:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy + 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set
        # 反 L 形
        if self.block_type == 7:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx + 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set

        return board_state + virtual_state[1:, :], virtual_state

    def move_left(self, board_state):
        """
        向左移动俄罗斯方块
        返回操作是否成功标志
        """
        tmpstate = self.get_preview(
            self.center_point[1] - 1, self.center_point[0], self.rotate
        )
        if np.any(tmpstate[:, 0:2] == init_value):
            # print("已经到达左边边界，无法继续往左移动")
            return False
        tmpboard = board_state + tmpstate[1 : row_count + 1, 2 : col_count + 2]
        if np.any(tmpboard == init_value + solid_value):
            # print("左边被方块挡住了，没法往左移动")
            return False

        self.center_point = (self.center_point[0], self.center_point[1] - 1)
        return True

    def move_right(self, board_state):
        """
        向右移动俄罗斯方块
        返回操作是否成功标志
        """
        tmpstate = self.get_preview(
            self.center_point[1] + 1, self.center_point[0], self.rotate
        )
        if np.any(tmpstate[:, -2:-1] == init_value):
            # print("已经到达右边边界，无法继续往右移动")
            return False
        tmpboard = board_state + tmpstate[1 : row_count + 1, 2 : col_count + 2]
        if np.any(tmpboard == init_value + solid_value):
            # print("右边被方块挡住了，没法往右移动")
            return False

        self.center_point = (self.center_point[0], self.center_point[1] + 1)
        return True

    def move_down(self, board_state):
        """
        向下移动不用预览
        返回当前俄罗斯方块是否死掉
        """
        tmpstate = self.get_preview(
            self.center_point[1], self.center_point[0] + 1, self.rotate
        )
        # 移动到底部了，这个方块死掉，生成一个新的方块
        if np.any(tmpstate[row_count + 1 :, :] == init_value):
            self.is_active = False
            # print("俄罗斯方块死亡 : 到达了底部后又继续向下移动")
            return True

        tmpboard = board_state + tmpstate[1 : row_count + 1, 2 : col_count + 2]
        # 往下移动的时候，被别的方块挡住了，这个方块死掉，生成一个新的方块
        if np.any(tmpboard == init_value + solid_value):
            self.is_active = False
            # print("俄罗斯方块死亡 : 被别的方块挡住又继续向下移动")
            return True

        self.center_point = (self.center_point[0] + 1, self.center_point[1])
        return False

    def move_bottom(self, board_state):
        """
        向下移动到可能到达的范围的最底部
        返回值：当前俄罗斯方块是否死亡，通常为True
        """
        tmpstep = 1
        while True:
            tmpstate = self.get_preview(
                self.center_point[1], self.center_point[0] + tmpstep, self.rotate
            )
            if np.any(tmpstate[row_count + 1 :, :] == init_value):
                self.is_active = False
                self.center_point = (
                    self.center_point[0] + tmpstep - 1,
                    self.center_point[1],
                )
                # print(f"{tmpstep - 1} , 俄罗斯方块死亡，到达了底部后又继续向下移动")
                return True

            tmpboard = board_state + tmpstate[1 : row_count + 1, 2 : col_count + 2]
            if np.any(tmpboard == init_value + solid_value):
                self.is_active = False
                self.center_point = (
                    self.center_point[0] + tmpstep - 1,
                    self.center_point[1],
                )
                # print(f"{tmpstep - 1} , 俄罗斯方块死亡，被别的方块挡住又继续向下移动")
                return True

            tmpstep += 1

            if tmpstep > row_count:
                raise Exception("")
                # print("移动到底部出错！！！")

        return False

    def rotate_block(self, board_state):
        """
        旋转俄罗斯方块
        返回操作是否成功标志
        """
        tmprotate = self.rotate
        # 一字形、z形、反z形的旋转
        if self.block_type == 1 or self.block_type == 4 or self.block_type == 5:
            if tmprotate == 0:
                tmprotate = 90
            else:
                tmprotate = 0
        # 正方块的旋转，总是返回False
        if self.block_type == 2:
            tmprotate = 0
            return False
        # 山形、L形、反L形的旋转
        if self.block_type == 3 or self.block_type == 6 or self.block_type == 7:
            # 旋转 90、180、270、0、90、180、270、0 ...
            if tmprotate < 270:
                tmprotate += 90
            else:
                tmprotate = 0

        tmpstate = self.get_preview(
            self.center_point[1], self.center_point[0], tmprotate
        )
        if np.any(tmpstate[:, 0:2] == init_value):
            # print("旋转失败，左边越界")
            return False
        if np.any(tmpstate[:, -2:-1] == init_value):
            # print("旋转失败，右边越界")
            return False
        if np.any(tmpstate[row_count + 1 :, :] == init_value):
            # print("旋转失败，底部越界")
            return False
        tmpboard = board_state + tmpstate[1 : row_count + 1, 2 : col_count + 2]
        if np.any(tmpboard == init_value + solid_value):
            # print("旋转失败，和别的方块重合")
            return False

        self.rotate = tmprotate
        return True

    def proc_when_block_dead(self, board_state):
        """
        当一个俄罗斯方块死掉的时候：
        1. 更新当前局面
        2. 计算可以清除的行、并更新局面
        3. 检查游戏是否结束
           更新后的局面是否有越界的俄罗斯方块
        4. 在顶部生成新的方块
        5. 检查游戏是否结束
           新生成的方块是否和已经死掉的方块重合
        返回
        1. 游戏是否结束
        2. 消除的行数 (0-4)
        3. 新的俄罗斯方块
           如果当前方块尚未死掉，返回空
        4. 最终的局面
           消除后行的局面、或者加入新方块的局面、结束的话返回空
        5. debug用的局面
        6. board局面
        """
        virtual_state = np.zeros((row_count + 1, col_count), np.ubyte)
        value_to_set = solid_value
        tmpx, tmpy = self.center_point[1], self.center_point[0] + 1

        # 横条
        if self.block_type == 1:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 3] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 3, tmpx] = value_to_set

        # 正方形
        if self.block_type == 2:
            virtual_state[tmpy : tmpy + 2, tmpx : tmpx + 2] = value_to_set

        # 山形
        if self.block_type == 3:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set

        # Z 形
        if self.block_type == 4:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 1] = value_to_set
                virtual_state[tmpy, tmpx : tmpx + 2] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1 : tmpy + 1, tmpx] = value_to_set

        # 反 Z 形
        if self.block_type == 5:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1 : tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 1, tmpx - 1] = value_to_set
                virtual_state[tmpy : tmpy + 2, tmpx] = value_to_set

        # L 形
        if self.block_type == 6:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy + 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set

        # 反 L 形
        if self.block_type == 7:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx + 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set

        # 方块越界、游戏结束
        # if np.any(virtual_state[0, :] > 0):
        #    print("Game End", virtual_state.shape)
        # dbgimg = create_image_from_state(virtual_state)
        # dbgimg = cv2.cvtColor(dbgimg, cv2.COLOR_BGR2RGB)
        # cv2.imshow("debug", dbgimg)
        # return None

        res_board = virtual_state
        res_board[1:, :] = board_state + virtual_state[1:, :]

        cp_dst = np.zeros((row_count + 1, col_count), np.ubyte)
        cp_idx = 0

        clear_line_count = 0
        game_stop_flag = False
        tmp_block = None

        # 逐行检查、查看是否有可以消去的行
        for i in range(res_board.shape[0]):
            if np.all(res_board[res_board.shape[0] - i - 1, :] == solid_value):
                clear_line_count += 1
            else:
                cp_dst[res_board.shape[0] - cp_idx - 1, :] = res_board[
                    res_board.shape[0] - i - 1, :
                ]
                cp_idx += 1

        # if clear_line_count > 0:
        #     print(f"{clear_line_count} line cleared ...")

        # 先消除行、再检查是否有越界的
        if np.any(cp_dst[0, :] > 0):
            game_stop_flag = True
            # print("游戏结束，因为消除可以消除的行之后，仍然有越界的方块")
            return (
                game_stop_flag,
                clear_line_count * reward_for_each_cleared_line
                if clear_line_count > 0
                else operation_succeed_but_cleared_zero_line_reward,
                tmp_block,
                cp_dst[1:, :],
                cp_dst,
                cp_dst[1:, :],
            )

        # 随机生成一个俄罗斯方块
        tmp_block = block()
        # 检查此随机生成的方块能否放在局面上
        tmp_game_state, block_state = tmp_block.preview_init(cp_dst[1:, :])

        # 检查新生成的方块是否和死掉的方块有重合
        if np.any(tmp_game_state == init_value + solid_value):
            game_stop_flag = True
            # print("游戏结束，因为没有空间容纳新的俄罗斯方块")
            return (
                game_stop_flag,
                clear_line_count * reward_for_each_cleared_line
                if clear_line_count > 0
                else operation_succeed_but_cleared_zero_line_reward,
                tmp_block,
                cp_dst[1:, :],
                block_state,
                cp_dst[1:, :],
            )

        return (
            game_stop_flag,
            clear_line_count * reward_for_each_cleared_line
            if clear_line_count > 0
            else operation_succeed_but_cleared_zero_line_reward,
            tmp_block,
            tmp_game_state,
            None,
            cp_dst[1:, :],
        )

    def get_preview(self, tmpx, tmpy, tmprotate):
        # 新坐标
        tmpx, tmpy = tmpx + 2, tmpy + 1
        # 新矩阵
        virtual_state = np.zeros((1 + row_count + 2, 2 + col_count + 2), np.ubyte)
        # 横条
        if self.block_type == 1:
            if tmprotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 3] = init_value
            if tmprotate == 90:
                virtual_state[tmpy - 1 : tmpy + 3, tmpx] = init_value
        # 正方形
        if self.block_type == 2:
            virtual_state[tmpy : tmpy + 2, tmpx : tmpx + 2] = init_value

        # 山形
        if self.block_type == 3:
            if tmprotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy - 1, tmpx] = init_value
            if tmprotate == 180:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy + 1, tmpx] = init_value
            if tmprotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = init_value
                virtual_state[tmpy, tmpx - 1] = init_value
            if tmprotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = init_value
                virtual_state[tmpy, tmpx + 1] = init_value

        # Z 形
        if self.block_type == 4:
            if tmprotate == 0:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 1] = init_value
                virtual_state[tmpy, tmpx : tmpx + 2] = init_value
            if tmprotate == 90:
                virtual_state[tmpy : tmpy + 2, tmpx - 1] = init_value
                virtual_state[tmpy - 1 : tmpy + 1, tmpx] = init_value

        # 反 Z 形
        if self.block_type == 5:
            if tmprotate == 0:
                virtual_state[tmpy - 1, tmpx : tmpx + 2] = init_value
                virtual_state[tmpy, tmpx - 1 : tmpx + 1] = init_value
            if tmprotate == 90:
                virtual_state[tmpy - 1 : tmpy + 1, tmpx - 1] = init_value
                virtual_state[tmpy : tmpy + 2, tmpx] = init_value

        # L 形
        if self.block_type == 6:
            if tmprotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy - 1, tmpx - 1] = init_value
            if tmprotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy, tmpx + 1] = init_value
            if tmprotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = init_value
                virtual_state[tmpy + 1, tmpx - 1] = init_value
            if tmprotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = init_value
                virtual_state[tmpy - 1, tmpx] = init_value

        # 反 L 形
        if self.block_type == 7:
            if tmprotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy - 1, tmpx + 1] = init_value
            if tmprotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = init_value
                virtual_state[tmpy, tmpx - 1] = init_value
            if tmprotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = init_value
                virtual_state[tmpy - 1, tmpx - 1] = init_value
            if tmprotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = init_value
                virtual_state[tmpy + 1, tmpx] = init_value

        return virtual_state

        # return current_state + virtual_state[1:, 2:-2]

    def get_state(self, board_state):
        virtual_state = np.zeros((row_count + 1, col_count), np.ubyte)
        value_to_set = init_value
        if self.is_active == False:
            value_to_set = solid_value
        tmpx, tmpy = self.center_point[1], self.center_point[0] + 1

        # 横条
        if self.block_type == 1:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 3] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 3, tmpx] = value_to_set

        # 正方形
        if self.block_type == 2:
            virtual_state[tmpy : tmpy + 2, tmpx : tmpx + 2] = value_to_set

        # 山形
        if self.block_type == 3:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set

        # Z 形
        if self.block_type == 4:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 1] = value_to_set
                virtual_state[tmpy, tmpx : tmpx + 2] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1 : tmpy + 1, tmpx] = value_to_set

        # 反 Z 形
        if self.block_type == 5:
            if self.rotate == 0:
                virtual_state[tmpy - 1, tmpx : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1 : tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 1, tmpx - 1] = value_to_set
                virtual_state[tmpy : tmpy + 2, tmpx] = value_to_set

        # L 形
        if self.block_type == 6:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx + 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy + 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy - 1, tmpx] = value_to_set

        # 反 L 形
        if self.block_type == 7:
            if self.rotate == 0:
                virtual_state[tmpy, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy - 1, tmpx + 1] = value_to_set
            if self.rotate == 180:
                virtual_state[tmpy - 1, tmpx - 1 : tmpx + 2] = value_to_set
                virtual_state[tmpy, tmpx - 1] = value_to_set
            if self.rotate == 90:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx] = value_to_set
                virtual_state[tmpy - 1, tmpx - 1] = value_to_set
            if self.rotate == 270:
                virtual_state[tmpy - 1 : tmpy + 2, tmpx - 1] = value_to_set
                virtual_state[tmpy + 1, tmpx] = value_to_set

        return board_state + virtual_state[1:, :]
