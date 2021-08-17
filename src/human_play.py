import cv2
from game.confs import Action_Type, Block_Type, Confs
import numpy as np
from game.tetris_engine import tetris_engine
from utils.util import create_image_from_state


def human_play():
    env = tetris_engine([Block_Type.L])
    game_state = env.reset()
    debug_img = None
    is_end = False
    while True:
        img = create_image_from_state(game_state)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape, img.dtype)
        # print(img)
        cv2.imshow("frame", img)
        if debug_img is not None:
            cv2.imshow("debug", debug_img)
        key = cv2.waitKey(10)
        # press Q or ESC
        if key == ord("q") or key == 27:
            break

        if is_end:
            continue

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
            # testcnn_reward(game_state)
            # print(f"reward [{reward}], is_end [{is_end}]")
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
    human_play()
