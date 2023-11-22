###${_STUB_IMPORTS} # noqa
from ..import_stub import *
from ..import_stub import node, IOOutput, IOInput
from typing import Dict, Literal, Union
proto: Dict[Literal[
    "processed_pd",
    "pre_lm_manip_cfg",
    "lm_nn_data",
    "hand_trace4_output_left",
    "hand_trace4_output_right",
    "early_out_lm",
], Union[IOOutput, IOInput]] = {}
node.io = proto
###${_STUB_IMPORTS} # noqa

script_name = ""  ###${_NAME} # noqa
fps = 0.0  ###${_fps} # noqa

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
    pd_detections: NNData = node.io['processed_pd'].get()
    palm_results = pd_detections.getLayerInt32('palm_results_processed')

    if len(palm_results) > 0:
        cfg: ImageManipConfig = ImageManipConfig()
        # TODO read pd_detections to determine cfg
        node.io['pre_lm_manip_cfg'].send(cfg)

        hand_lms: NNData = node.io['lm_nn_data'].get()
        """
        hand_lms = node.io["lm_nn_data"].tryGet()
        if hand_lms:
            hand_frame = node.io["lm_nn_frame"].get()
        ###${_TRACE2}node.warn(f"{script_name} received result from lm nn")${_TRACE2} # noqa
        lm_score = hand_lms.getLayerFp16("Identity_1")[0]
        if lm_score > lm_score_thresh:
            # noinspection DuplicatedCode
            handedness = hand_lms.getLayerFp16("Identity_2")[0]
            rrn_lms = hand_lms.getLayerFp16("Identity_dense/BiasAdd/Add")
            world_lms = hand_lms.getLayerFp16("Identity_3_dense/BiasAdd/Add")
        """
        ###${_TRACE4} # noqa

        # TODO
        node.io['hand_trace4_output_left'].send()
        node.io['hand_trace4_output_right'].send()
        ###${_TRACE4} # noqa
    else:
        node.io['early_out_lm'].send(pd_detections)

    nb_hands_in_previous_frame = 0
