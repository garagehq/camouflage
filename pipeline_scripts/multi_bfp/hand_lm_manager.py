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
    pd_detections: NNData = node.io['processed_pd'].get()

    if pd_detections.hasLayer('result_processed'):
        cfg: ImageManipConfig = ImageManipConfig()
        # TODO read pd_detections to determine cfg
        # noinspection PyUnresolvedReferences
        node.io['pre_lm_manip_cfg'].send(cfg)

        hand_lms: NNData = node.io['lm_nn_data'].get()
        """
        hand_lms = node.io["lm_nn_data"].tryGet()
        if hand_lms:
            # Docs guarantee you can do this using passthrough(out<->passthrough are 1 to 1)
            # noinspection PyUnresolvedReferences
            hand_frame = node.io["lm_nn_frame"].get()
        ###${_TRACE2}(f"${_NAME} received result from lm nn") # noqa
        lm_score = hand_lms.getLayerFp16("Identity_1")[0]
        if lm_score > lm_score_thresh:
            # noinspection DuplicatedCode
            handedness = hand_lms.getLayerFp16("Identity_2")[0]
            rrn_lms = hand_lms.getLayerFp16("Identity_dense/BiasAdd/Add")
            world_lms = 0  # ${_IF_USE_WORLD_LANDMARKS}##hand_lms.getLayerFp16("Identity_3_dense/BiasAdd/Add")${_IF_USE_WORLD_LANDMARKS} # noqa
        """
    else:
        # noinspection PyUnresolvedReferences
        node.io['early_out_lm'].send(frame)

    # noinspection PyUnresolvedReferences
    node.io['early_out_lm'].send(frame)

    nb_hands_in_previous_frame = 0
