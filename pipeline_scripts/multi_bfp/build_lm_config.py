import time

###${_STUB_IMPORTS} # noqa
from depthai import *
###${_STUB_IMPORTS} # noqa

fps = 0.0  ###${_fps}# noqa

pad_h = 0  ###${_pad_h} # noqa
img_h = 0  ###${_img_h} # noqa
img_w = 0  ###${_img_w} # noqa
frame_size = 0  ###${_frame_size} # noqa

# For Movenet smart cropping, defines the default crop region (pads the full image from both sides to make it a
# square image) Used when the algorithm cannot reliably determine the crop region from the previous frame.
init_crop_region = {'x_min': 0, 'y_min': -pad_h, 'x_max': frame_size, 'y_max': -pad_h + frame_size, 'size': frame_size}
# noinspection DuplicatedCode
crop_region = init_crop_region

body_score_thresh = 0.0  ###${_body_score_thresh}# noqa
hands_up_only = False  ###${_hands_up_only}# noqa

pd_score_thresh = 0.0  ###${_pd_score_thresh}# noqa
lm_score_thresh = 0.0  ###${_lm_score_thresh}# noqa
single_hand_tolerance_thresh = 0.0  ###${_single_hand_tolerance_thresh}# noqa

# noinspection DuplicatedCode
BODY_KP = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

torso_joints = [BODY_KP["left_hip"], BODY_KP["right_hip"], BODY_KP["left_shoulder"], BODY_KP["right_shoulder"]]


while True:
    # noinspection PyUnresolvedReferences
    pd_detections = node.io['pd_data'].get().getLayerFp16("result")
    # noinspection PyUnresolvedReferences
    frame = node.io['pd_frame'].get()
    ###${_TRACE_DUMP}(f"${_NAME}: processing frame {frame.getSequenceNum()}") # noqa

    pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = detection[i * 8:(i + 1) * 8]
    if pd_score >= pd_score_thresh and box_size > 0:
        # noinspection DuplicatedCode  # noqa
        if zone:
            # x_min, y_min, x_max are expressed in pixel in the source image C.S. box_x, box_y, box_size,
            # kp0_x, kp0_y, kp2_x, kp2_y are normalized coords in square zone We need box_x, box_y, box_size,
            # kp0_x, kp0_y, kp2_x, kp2_y expressed in normalized coords in squared source image (sqn_)!
            # conversion factor from frame size to box size
            sqn_zone_size = (x_max - x_min) / frame_size
            # determine box size relative to frame in normals
            box_size *= sqn_zone_size
            # divide by conversion factor for box norms -> frame norms
            sqn_x_min = x_min / frame_size
            sqn_y_min = (y_min + pad_h) / frame_size

            box_x = (box_x * sqn_zone_size) + sqn_x_min
            # node.warn(f"{round(sqn_y_min, 2)}, {round(box_y, 2)}, {round(sqn_zone_size, 2)}")
            box_y = (box_y * sqn_zone_size) + sqn_y_min
            kp0_x = (kp0_x * sqn_zone_size) + sqn_x_min
            kp0_y = (kp0_y * sqn_zone_size) + sqn_y_min
            kp2_x = (kp2_x * sqn_zone_size) + sqn_x_min
            kp2_y = (kp2_y * sqn_zone_size) + sqn_y_min
            out.append(box_x)

        # scale_center_x = sqn_scale_x - sqn_rr_center_x
        # scale_center_y = sqn_scale_y - sqn_rr_center_y
        kp02_x = kp2_x - kp0_x
        kp02_y = kp2_y - kp0_y
        sqn_rr_size = 2.9 * box_size
        rotation = 0.5 * pi - atan2(-kp02_y, kp02_x)
        rotation = normalize_radians(rotation)
        sqn_rr_center_x = box_x + 0.5 * box_size * sin(rotation)
        sqn_rr_center_y = box_y - 0.5 * box_size * cos(rotation)
        hands.append([sqn_rr_size, rotation, sqn_rr_center_x, sqn_rr_center_y])
        node.io['pre_lm_manip_cfg'].send(frame)
    else:
        # noinspection PyUnresolvedReferences
        node.io['early_out_lm'].send(frame)

    nb_hands_in_previous_frame = 0

