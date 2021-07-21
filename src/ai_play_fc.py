"""
此脚本负责 - 手工运行游戏或者用AI运行游戏
"""
import sys
import cv2
import torch
from model.cnn_model import DQN
from model.fc_model import DQN_FC
from utils.util import create_image_from_state
from game.confs import Action_Type, Block_Type, Confs
from game.tetris_engine import tetris_engine


def ai_play(model_file):
    model = DQN_FC(Confs.row_count.value + 1, Confs.col_count.value, 4)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    env = tetris_engine()
    game_state = env.reset()
    debug_img = None
    is_end = False
    while True:
        img = create_image_from_state(game_state)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        key = cv2.waitKey(10)
        # press Q or ESC
        if key == ord("q") or key == 27:
            break

        if is_end:
            continue

        tensor = torch.from_numpy(game_state.flatten()).float().unsqueeze(0)
        pred_q = model(tensor)
        # print("##############")
        # print(pred_q)
        lt, rt, r, d = (
            pred_q.data[0][0].item(),
            pred_q.data[0][1].item(),
            pred_q.data[0][2].item(),
            pred_q.data[0][3].item(),
        )
        # print(pred_q.data)
        # print(pred_q.data.max(1))
        # print(pred_q.data.max(1)[1])
        # print(pred_q.data.max(1)[1].view(1, 1))
        # print(pred_q.data.max(1)[1].item())
        # select_idx = pred_q.data.max(1)[1].item()
        # if select_idx == 0:
        #     k = ord("a")
        #     print("Model suggest left action")
        # elif select_idx == 1:
        #     k = ord("d")
        #     print("Model suggest right action")
        # elif select_idx == 2:
        #     k = ord("w")
        #     print("Model suggest rotate action")
        # elif select_idx == 3:
        #     k = ord("s")
        #     print("Model suggest down action")
        # else:
        #     raise Exception("Error prediction")

        print(
            [
                env.test_step(Action_Type.Left) - lt,
                env.test_step(Action_Type.Right) - rt,
                env.test_step(Action_Type.Rotate) - r,
                env.test_step(Action_Type.Down) - d,
            ]
        )
        # print(
        #     f"left max step is {env.test_step(Action_Type.Left)} , right max step is {env.test_step(Action_Type.Right)} , down max step is {env.test_step(Action_Type.Down)} , rotate max step is {env.test_step(Action_Type.Rotate)}"
        # )

        if key == ord("w"):
            # rotate
            game_state, reward, is_end, debug = env.step(Action_Type.Rotate)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("s"):
            # down
            game_state, reward, is_end, debug = env.step(Action_Type.Down)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("a"):
            # left
            game_state, reward, is_end, debug = env.step(Action_Type.Left)
            # print(f"reward [{reward}], is_end [{is_end}]")
            if debug is not None:
                debug_img = create_image_from_state(debug)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        elif key == ord("d"):
            # right
            game_state, reward, is_end, debug = env.step(Action_Type.Right)
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
    ai_play("outputs/Tetris_FC_400000.pt")
    sys.exit(0)
