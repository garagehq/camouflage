import time
from math import dist

###${_STUB_IMPORTS} # noqa
from depthai import *

###${_STUB_IMPORTS} # noqa

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


def centernet_postprocess(body_lms):
    scores = []
    x = []
    y = []
    for i in range(17):
        xn = body_lms[3 * i + 1]
        yn = body_lms[3 * i]
        scores.append(body_lms[3 * i + 2])
        # x any are the body keypoint coordinates in the source image in pixels
        x.append(int(crop_region['x_min'] + xn * crop_region['size']))
        y.append(int(crop_region['y_min'] + yn * crop_region['size']))

    return x, y, scores


# noinspection DuplicatedCode
def get_focus_zone(p_body_x, p_body_y, p_body_scores):
    #  Return a list [focus_zone, label]
    # - 'focus_zone' is a zone around a hand or hands, depending on the value
    #   of hand_label (`left`, `right`, `higher` or `group`) and on the value of self.hands_up_only.
    #       - hand_label = `left` (resp `right`): we are looking for the zone around the left (resp right) wrist,
    #       - hand_label = `group`: the zone encompasses both wrists,
    #       - hand_label = `higher`: the zone is around the higher wrist (smaller y value),
    #       - hands_up_only = True: we don't consider the wrist if the corresponding elbow is above the wrist,
    #   focus_zone is a list [left, top, right, bottom] defining the top-left and right-bottom corners of a square.
    #   Values are expressed in pixels in the source image C.S.
    #   The zone is constrained to the squared source image (= source image with padding if necessary).
    #   It means that values can be negative.
    #   left and right in [-pad_w, img_w + pad_w]
    #   top and bottom in [-pad_h, img_h + pad_h]
    # - 'label' describes which wrist keypoint(s) were used to build the zone : `left`, `right` or `group`
    #   (if built from both wrists)
    # If the wrist keypoint(s) is(are) not present or is(are) present but self.hands_up_only = True and
    # wrist(s) is(are) below corresponding elbow(s), then focus_zone = None.

    # noinspection DuplicatedCode
    def zone_from_center_size(zone_x, zone_y, zone_size):
        # Return zone [left, top, right, bottom]
        # from zone center (x,y) and zone size (the zone is square).
        half_size = zone_size // 2
        if zone_size > img_w:
            zone_x = img_w // 2
        x1 = zone_x - half_size
        if x1 < 0:
            x1 = 0
        elif x1 + zone_size > img_w:
            x1 = img_w - zone_size
        x2 = x1 + zone_size
        if zone_size > img_h:
            zone_y = img_h // 2
        y1 = zone_y - half_size
        if y1 < -pad_h:
            y1 = -pad_h
        elif y1 + zone_size > img_h + pad_h:
            y1 = img_h + pad_h - zone_size
        y2 = y1 + zone_size
        return [x1, y1, x2, y2]

    def get_one_hand_zone(hand_label_to_eval):
        # Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
        # Values are expressed in pixels in the source image C.S.
        # If the wrist keypoint is not visible, return None.
        # If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.

        # noinspection DuplicatedCode
        id_wrist = BODY_KP[hand_label_to_eval + "_wrist"]
        if p_body_scores[id_wrist] < body_score_thresh / 2:
            return None
        wrist_x = p_body_x[id_wrist]
        wrist_y = p_body_y[id_wrist]
        if hands_up_only:
            # TODO reimplement this based on the xyz readings rather than 2D pixel readings, the overhead perspective
            #  isn't compatible with the assumptions of this algorithm
            # We want to detect only hands where the wrist is above the hip (when visible)
            elbow_kp = hand_label_to_eval + "_hip"
            id_elbow = BODY_KP[elbow_kp]
            if p_body_scores[id_elbow] > body_score_thresh and \
                    p_body_y[id_elbow] < wrist_y:
                return None

        # Let's evaluate the size of the focus zone

        # noinspection DuplicatedCode
        def estimate_focus_zone_size(scale=1.0):
            segments = [
                ("left_shoulder", "left_elbow", 2.3),
                ("left_elbow", "left_wrist", 2.3),
                ("left_shoulder", "left_hip", 1),
                ("left_shoulder", "right_shoulder", 1.5),
                ("right_shoulder", "right_elbow", 2.3),
                ("right_elbow", "right_wrist", 2.3),
                ("right_shoulder", "right_hip", 1),
            ]
            lengths = []
            for s in segments:
                id0 = BODY_KP[s[0]]
                id1 = BODY_KP[s[1]]
                if p_body_scores[id0] > body_score_thresh and p_body_scores[id1] > body_score_thresh:
                    weight_l = dist((p_body_x[id0], p_body_y[id0]), (p_body_x[id1], p_body_y[id1]))
                    lengths.append(weight_l)
            if lengths:
                if (p_body_scores[BODY_KP["left_hip"]] < body_score_thresh and
                        p_body_scores[BODY_KP["right_hip"]] < body_score_thresh or
                        p_body_scores[BODY_KP["left_shoulder"]] < body_score_thresh and
                        p_body_scores[BODY_KP["right_shoulder"]] < body_score_thresh):
                    coefficient = 1.0
                else:
                    coefficient = 1.0
                return 2 * int(coefficient * scale * max(lengths) / 2)  # The size is made even
                # return coefficient * scale * max(lengths)
            else:
                return 0

        focus_size = estimate_focus_zone_size()
        ###${_TRACE_DUMP}(f"${_NAME} focus zone size????? {focus_size} center: ({wrist_x},{wrist_y})") # noqa
        return [0, -pad_h, frame_size, frame_size - pad_h] if focus_size == 0 else zone_from_center_size(wrist_x,
                                                                                                         wrist_y, focus_size)

    zone_l = get_one_hand_zone("left")
    zone_r = get_one_hand_zone("right")
    ###${_TRACE_DUMP}(f"${_NAME} hand zones: l: {zone_l} r: {zone_r}") # noqa
    if zone_l is not None and zone_r is not None:
        xl1, yl1, xl2, yl2 = zone_l
        xr1, yr1, xr2, yr2 = zone_r
        x1_lr = min(xl1, xr1)
        y1_lr = min(yl1, yr1)
        x2_lr = max(xl2, xr2)
        y2_lr = max(yl2, yr2)
        # Center (x,y)
        combined_center_x = (x1_lr + x2_lr) // 2
        combined_center_y = (y1_lr + y2_lr) // 2
        size_x = x2_lr - x1_lr
        size_y = y2_lr - y1_lr
        size = max(size_x, size_y)
        zone_hand = [zone_from_center_size(combined_center_x, combined_center_y, size), "group"]
    elif zone_l is None and zone_r is None:
        zone_hand = [None, None]
    else:
        zone_hand = [zone_l, "left"] if zone_l is not None else [zone_r, "right"]
    ###${_TRACE_DUMP}(f"${_NAME} hand zone determination: {zone_hand}") # noqa
    return zone_hand


def normalize_results(results, detection_zone):
    # TODO
    """
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
    """
    return [0]


while True:
    # noinspection PyUnresolvedReferences
    body = node.io['body_nn_data'].get().getLayerFp16("Identity")
    # noinspection PyUnresolvedReferences
    frame: ImgFrame = node.io['body_nn_frame'].get()
    ###${_TRACE_DUMP}(f"${_NAME} processing frame {frame.getSequenceNum()}") # noqa

    # Extract body keypoints TODO many bodies
    body_x, body_y, body_scores = centernet_postprocess(body)

    ###${_TRACE_DUMP}(f"${_NAME} got body keypoints") # noqa

    # Calculate pre focus zone
    zone, hand_label = get_focus_zone(body_x, body_y, body_scores)

    ###${_TRACE_DUMP}(f"${_NAME} focus zone calculation complete") # noqa

    if zone:
        # noinspection DuplicatedCode
        x_min, y_min, x_max, y_max = zone
        ###${_TRACE1}(f"Body pre focusing zone: ({x_min}, {y_min}), ({x_max}, {y_max})") # noqa
        # noinspection DuplicatedCode
        points = [
            Point2f(x_min, y_min),
            Point2f(x_max, y_min),
            Point2f(x_max, y_max),
            Point2f(x_min, y_max)
        ]
        cfg_pre_pd = ImageManipConfig()
        cfg_pre_pd.setWarpTransformFourPoints(points, False)
        cfg_pre_pd.setResize(128, 128)
        # noinspection PyUnresolvedReferences
        node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)

        # noinspection PyUnresolvedReferences
        nn_data: NNData = node.io['pd_data'].get()
        # TODO may need to set seqNumber from the frame above if the NN node doesn't
        # pd_detections = nn_data.getLayerFp16("result")
        normalized_palms = normalize_results(nn_data.getLayerFp16("result"), zone)
        nn_data.setLayer("result_processed", normalized_palms)
        # noinspection PyUnresolvedReferences
        node.io['processed_pd'].send(nn_data)
    else:
        ###${_TRACE1}(f"Body pre focusing zone: None") # noqa
        # noinspection PyUnresolvedReferences
        node.io['early_out_pd'].send(frame)

