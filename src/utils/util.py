import numpy as np


def create_image_from_state(state: np.ndarray):
    tmp_col_count = state.shape[1]
    tmp_row_count = state.shape[0]
    box_center_dist = 30
    box_width = 13
    centers_x = [_ * box_center_dist + box_width for _ in range(tmp_col_count)]
    centers_y = [_ * box_center_dist + box_width for _ in range(tmp_row_count)]
    bkg = np.full(
        (centers_y[-1] + box_width, centers_x[-1] + box_width, 3),
        (255, 255, 255),
        np.uint8,
    )
    for j in range(tmp_row_count):
        for i in range(tmp_col_count):
            if state[j, i] == 0:
                bkg[
                    centers_y[j] - box_width : centers_y[j] + box_width + 1,
                    centers_x[i] - box_width : centers_x[i] + box_width + 1,
                ] = (235, 235, 235)
            elif state[j, i] == 1:
                bkg[
                    centers_y[j] - box_width : centers_y[j] + box_width + 1,
                    centers_x[i] - box_width : centers_x[i] + box_width + 1,
                ] = (0, 0, 0)
            else:
                bkg[
                    centers_y[j] - box_width : centers_y[j] + box_width + 1,
                    centers_x[i] - box_width : centers_x[i] + box_width + 1,
                ] = (255, 0, 0)
    return bkg
