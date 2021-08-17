"""
此脚本负责 - 手工运行游戏或者用AI运行游戏
"""
import sys
import cv2
import torch
from model.cnn_model import DQN
from utils.util import create_image_from_state
from game.confs import Action_Type, Block_Type, Confs
from game.tetris_engine import tetris_engine


def ai_play(model_file):
    model = DQN(Confs.row_count.value + 1, Confs.col_count.value, 3)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    env = tetris_engine([Block_Type.O])
    game_state = env.reset()
    debug_img = None
    is_end = False
    while True:
        img = create_image_from_state(game_state)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        key = cv2.waitKey(20)
        # press Q or ESC
        if key == ord("q") or key == 27:
            break

        if is_end:
            continue

        tensor = torch.from_numpy(game_state).unsqueeze(0).unsqueeze(0).float()
        pred_q = model(tensor)
        # print("##############")
        # print(pred_q.data)
        # print(pred_q.data.max(1))
        # print(pred_q.data.max(1)[1])
        # print(pred_q.data.max(1)[1].view(1, 1))
        # print(pred_q.data.max(1)[1].item())
        select_idx = pred_q.data.max(1)[1].item()
        if select_idx == 0:
            key = ord("a")
            print("left", pred_q.data)
        elif select_idx == 1:
            key = ord("d")
            print("right", pred_q.data)
        elif select_idx == 2:
            key = ord("s")
            print("down", pred_q.data)
        elif select_idx == 3:
            key = ord("w")
            print("error", pred_q.data)
        else:
            raise Exception("Error prediction")

        # print(
        #     f"left max step is {env.test_step(Action_Type.Left)} , right max step is {env.test_step(Action_Type.Right)} , down max step is {env.test_step(Action_Type.Down)} , rotate max step is {env.test_step(Action_Type.Rotate)}"
        # )

        if key == ord("w"):
            # rotate
            game_state, reward, is_end, debug = env.step(Action_Type.Rotate_Down)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("s"):
            # down
            game_state, reward, is_end, debug = env.step(Action_Type.Down)
            # print(f"reward [{reward}], is_end [{is_end}]")
            # print(game_state)
            # print(debug)
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("a"):
            # left
            game_state, reward, is_end, debug = env.step(Action_Type.Left_Down)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("d"):
            # right
            game_state, reward, is_end, debug = env.step(Action_Type.Right_Down)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord(" "):
            # bottom
            game_state, reward, is_end, debug = env.step(Action_Type.Bottom)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # human_play()
    ai_play("outputs/Tetris_8000000-124.pt")
    sys.exit(0)
