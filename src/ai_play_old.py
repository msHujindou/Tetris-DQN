"""
此脚本负责 - 手工运行游戏或者用AI运行游戏
"""
import sys

import numpy as np
import cv2
import torch

from game.tetris import block
from model.cnn_model import DQN
from utils.util import create_image_from_state

# 俄罗斯方块矩阵的高度和宽度
row_count, col_count = 20, 10


def ai_play(model_file):
    # 检查游戏是否成功
    board_state = np.zeros((row_count, col_count), np.ubyte)
    game_block = block()
    game_state, bloc_state = game_block.preview_init(board_state)

    action_space = [0, 1, 2, 3]
    if model_file is not None:
        model = DQN(row_count, col_count, len(action_space))
        model.load_state_dict(torch.load(model_file))
        model.eval()

    stopflag = False
    debug_img = None
    while True:
        img = create_image_from_state(game_state)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        k = cv2.waitKey(50)
        # press Q or ESC
        if k == ord("q") or k == 27:
            break

        if stopflag:
            continue

        if model_file is not None:
            tensor = torch.from_numpy(game_state)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.unsqueeze(0)
            tensor = tensor.float()
            pred_q = model(tensor)
            # print("##############")
            print(pred_q)
            # print(pred_q.data)
            # print(pred_q.data.max(1))
            # print(pred_q.data.max(1)[1])
            # print(pred_q.data.max(1)[1].view(1, 1))
            # print(pred_q.data.max(1)[1].item())

            select_idx = pred_q.data.max(1)[1].item()
            if select_idx == 0:
                k = ord("a")
                print("Model suggest left action")
            elif select_idx == 1:
                k = ord("d")
                print("Model suggest right action")
            elif select_idx == 2:
                k = ord("w")
                print("Model suggest rotate action")
            elif select_idx == 3:
                k = ord("s")
                print("Model suggest down action")
            else:
                raise Exception("Error prediction")

        if k == ord("w"):
            (
                stopflag,
                cleared_count,
                created_block,
                restate,
                dbginfo,
                board_state,
            ) = game_block.step(2, board_state)
            if restate is not None:
                game_state = restate
            if created_block is not None:
                game_block = created_block
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
            ) = game_block.step(3, board_state)
            if restate is not None:
                game_state = restate
            if created_block is not None:
                game_block = created_block
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
            ) = game_block.step(0, board_state)
            if restate is not None:
                game_state = restate
            # print(restate)
            if created_block is not None:
                game_block = created_block
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
            ) = game_block.step(1, board_state)
            if restate is not None:
                game_state = restate
            if created_block is not None:
                game_block = created_block
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
            ) = game_block.step(4, board_state)
            if restate is not None:
                game_state = restate
            if created_block is not None:
                game_block = created_block
            if dbginfo is not None:
                debug_img = create_image_from_state(dbginfo)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # human_play()
    ai_play("outputs/Tetris_10.pt")
    sys.exit(0)
