"""
此脚本负责 - 手工运行游戏或者用AI运行游戏
"""
import sys

import numpy as np
import cv2

from game.tetris import block
from utils.util import create_image_from_state

# 俄罗斯方块矩阵的高度和宽度
row_count, col_count = 20, 10


def human_play():
    # 检查游戏是否成功
    board_state = np.zeros((row_count, col_count), np.ubyte)
    line = block()
    game_state, bloc_state = line.preview_init(board_state)
    stopflag = False
    debug_img = None
    while True:
        img = create_image_from_state(game_state)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        k = cv2.waitKey(10)
        # press Q or ESC
        if k == ord("q") or k == 27:
            break

        if stopflag:
            continue

        if k == ord("w"):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = line.step(2, board_state)
            print(
                f"action:{'rotate'} ,new_game_state_returned:{restate is not None} ,new_tetris_block_created:{created_block is not None} ,gamestop:{stopflag}"
            )
            if restate is not None:
                game_state = restate
            if created_block is not None:
                line = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif k == ord("s"):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = line.step(3, board_state)
            print(
                f"action:{'down'} ,new_game_state_returned:{restate is not None} ,new_tetris_block_created:{created_block is not None} ,gamestop:{stopflag}"
            )
            if restate is not None:
                game_state = restate
            if created_block is not None:
                line = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif k == ord("a"):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = line.step(0, board_state)
            print(
                f"action:{'left'} ,new_game_state_returned:{restate is not None} ,new_tetris_block_created:{created_block is not None} ,gamestop:{stopflag}"
            )
            if restate is not None:
                game_state = restate
            if created_block is not None:
                line = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif k == ord("d"):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = line.step(1, board_state)
            print(
                f"action:{'right'} ,new_game_state_returned:{restate is not None} ,new_tetris_block_created:{created_block is not None} ,gamestop:{stopflag}"
            )
            if restate is not None:
                game_state = restate
            if created_block is not None:
                line = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif k == ord(" "):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = line.step(4, board_state)
            print(
                f"action:{'bottom'} ,new_game_state_returned:{restate is not None} ,new_tetris_block_created:{created_block is not None} ,gamestop:{stopflag}"
            )
            if restate is not None:
                game_state = restate
            if created_block is not None:
                line = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    human_play()
    sys.exit(0)
