import marshal
from math import dist, pi, sin, atan2, floor, degrees, cos, radians

###${_STUB_IMPORTS} # noqa
from ..import_stub import *
from ..import_stub import node

###${_STUB_IMPORTS} # noqa

script_name = ""  ###${_NAME} # noqa
fps = 0.0  ###${_fps} # noqa

pad_h = 0  ###${_pad_h} # noqa
img_h = 0  ###${_img_h} # noqa
img_w = 0  ###${_img_w} # noqa
frame_size = 0  ###${_frame_size} # noqa

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


def clamp_n(n):
    return min(1.0, max(0.0, n))


def clamp_px(n):
    return min(frame_size, max(0, n))


def centernet_postprocess(body_lms):
    # len(results) = 39
    # results == [b_box_x1, b_box_y1, b_box_x2, b_box_x2, confidence, kp_1_x, kp_1_y...]

    ###${_TRACE_INFO}node.warn(f"{script_name} movenet kps shape: {np.array(body_lms).shape}")###${_TRACE_INFO} # noqa

    viable_bodies = [body_res for body_res in body_lms if body_res[4] > body_score_thresh]

    ###${_TRACE_INFO}node.warn(f"{script_name} {len(viable_bodies)} bodies found")###${_TRACE_INFO} # noqa

    x = []
    y = []
    bboxes = []
    for current_body in viable_bodies:
        bboxes.append([point_half for point_half in current_body[:4]])
        x.append([x for x in current_body[5::2]])
        y.append([y for y in current_body[6::2]])

    return x, y, bboxes


# noinspection DuplicatedCode
def get_focus_zones(p_bodies_x, p_bodies_y, bounding_boxes):
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
    ret_zones = []
    for body_x_kps, body_y_kps, body_bounding_box in zip(p_bodies_x, p_bodies_y, bounding_boxes):
        # noinspection DuplicatedCode
        def zone_from_center_size_norm(zone_x_norm, zone_y_norm, zone_size_norm):
            # Return zone [left, top, right, bottom]
            # from zone center (x,y) and zone size (the zone is square).
            zone_size = zone_size_norm * frame_size
            half_size = zone_size // 2
            zone_x = img_w // 2 if zone_size_norm > 1.0 else zone_x_norm * frame_size
            zone_y = img_h // 2 if zone_size_norm > 1.0 else zone_y_norm * frame_size
            x1 = min(img_w - zone_size, max(0, zone_x - half_size))
            y1 = min(img_h + pad_h - zone_size, max(-pad_h, zone_y - half_size))
            ###${_TRACE_INFO}node.warn(f"{script_name} generated  {[x1, y1, x1 + zone_size, y1 + zone_size]}")${_TRACE_INFO} # noqa
            return [x1, y1, x1 + zone_size, y1 + zone_size]

        # def get_one_hand_zone(hand_label_to_eval):
        #     # Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
        #     # Values are expressed in pixels in the source image C.S.
        #     # If the wrist keypoint is not visible, return None.
        #     # If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
        #
        #     # noinspection DuplicatedCode
        #     id_wrist = BODY_KP[hand_label_to_eval + "_wrist"]
        #     wrist_x = p_body_x[id_wrist]
        #     wrist_y = p_body_y[id_wrist]
        #     # TODO reimplement this based on the xyz readings rather than 2D pixel readings, the overhead perspective
        #     #  isn't compatible with the assumptions of this algorithm
        #     # We want to detect only hands where the wrist is above the hip (when visible)
        #     id_hip = BODY_KP[hand_label_to_eval + "_hip"]
        #     if p_body_y[id_hip] < wrist_y and False:
        #         return None
        #
        #     ###${_TRACE_INFO}node.warn(f"{script_name} focus zone size????? {focus_size} center: ({wrist_x},{wrist_y})")${_TRACE_INFO} # noqa
        #     return [0, -pad_h, frame_size, frame_size - pad_h] \
        #         if focus_size == 0 \
        #         else zone_from_center_size(wrist_x, wrist_y, focus_size)
        # zone_l = get_one_hand_zone("left")
        # zone_r = get_one_hand_zone("right")
        # ###${_TRACE_INFO}node.warn(f"{script_name} hand zones: l: {zone_l} r: {zone_r}")${_TRACE_INFO} # noqa
        # if zone_l is not None and zone_r is not None:
        #     xl1, yl1, xl2, yl2 = zone_l
        #     xr1, yr1, xr2, yr2 = zone_r
        #     x1_lr = min(xl1, xr1)
        #     y1_lr = min(yl1, yr1)
        #     x2_lr = max(xl2, xr2)
        #     y2_lr = max(yl2, yr2)
        #     # Center (x,y)
        #     combined_center_x = (x1_lr + x2_lr) // 2
        #     combined_center_y = (y1_lr + y2_lr) // 2
        #     size_x = x2_lr - x1_lr
        #     size_y = y2_lr - y1_lr
        #     size = max(size_x, size_y)
        #     zone_hand = [zone_from_center_size(combined_center_x, combined_center_y, size), "group"]
        # elif zone_l is None and zone_r is None:
        #     zone_hand = []
        # else:
        #     zone_hand = [zone_l, "left"] if zone_l is not None else [zone_r, "right"]
        # ###${_TRACE_INFO}node.warn(f"{script_name} hand zone determination: {zone_hand}")${_TRACE_INFO} # noqa
        # xl1, yl1, xl2, yl2 = zone_l
        #     xr1, yr1, xr2, yr2 = zone_r
        #     x1_lr = min(xl1, xr1)
        #     y1_lr = min(yl1, yr1)
        #     x2_lr = max(xl2, xr2)
        #     y2_lr = max(yl2, yr2)
        #     # Center (x,y)
        #     combined_center_x = (x1_lr + x2_lr) // 2
        #     combined_center_y = (y1_lr + y2_lr) // 2
        #     size_x = x2_lr - x1_lr
        #     size_y = y2_lr - y1_lr
        #     size = max(size_x, size_y)
        #     zone_hand = [zone_from_center_size(combined_center_x, combined_center_y, size), "group"]

        # # x1, y1, x2, y2 of bounding box
        b_left, b_top, b_right, b_bottom, conf = body_bounding_box
        # wrist indexes
        l_wrist_idx, r_wrist_idx = BODY_KP["left_wrist"], BODY_KP["right_wrist"]
        # wrist normalized coordinates
        l_wrist = {'x': body_x_kps[l_wrist_idx], 'y': body_y_kps[l_wrist_idx]}
        r_wrist = {'x': body_x_kps[r_wrist_idx], 'y': body_y_kps[r_wrist_idx]}
        # # closest point to each of 4 sides
        # norm_to_bbox_l: float = min(l_wrist['x'] - b_left, r_wrist['x'] - b_left)
        # norm_to_bbox_r: float = min(b_right - l_wrist['x'], b_right - r_wrist['x'])
        # norm_to_bbox_t: float = min(l_wrist['y'] - b_top, l_wrist['y'] - b_top)
        # norm_to_bbox_b: float = min(b_bottom - l_wrist['y'], b_bottom - l_wrist['y'])
        #
        # # if the distance from keypoint to edge is less than (20%) of the box width/height,
        # #   then increase the size by (10%), else decrease it by (10%)
        # norm_px_thresh = .2
        # norm_width = b_right - b_left
        # norm_height = b_bottom - b_top
        #
        # # push or pull each side based on distance to wrist keypoints
        # b_left += norm_px_thresh * (-.5 if norm_to_bbox_l < norm_width * norm_px_thresh else .5)
        # b_top += norm_px_thresh * (-.5 if norm_to_bbox_r < norm_height * norm_px_thresh else .5)
        # b_right += norm_px_thresh * (.5 if norm_to_bbox_t < norm_width * norm_px_thresh else -.5)
        # b_bottom += norm_px_thresh * (.5 if norm_to_bbox_b < norm_height * norm_px_thresh else -.5)
        #
        # # clamp
        # b_left = max(b_left, 0.0)
        # b_top = max(b_top, 0.0)
        # b_right = min(b_right, 1.0)
        # b_bottom = min(b_bottom, 1.0)

        # rr between palms alg

        bbox_width_px = (b_right - b_left) * frame_size
        bbox_height_px = ((b_bottom - b_top) * frame_size) - pad_h

        l_wrist_px = {kp: int(norm * frame_size) - (0 if kp == 'x' else pad_h) for kp, norm in l_wrist.items()}
        r_wrist_px = {kp: int(norm * frame_size) - (0 if kp == 'x' else pad_h) for kp, norm in r_wrist.items()}

        x_diff, y_diff = l_wrist_px['x'] - r_wrist_px['x'], l_wrist_px['y'] - r_wrist_px['y']

        center_x, center_y = (l_wrist_px['x'] + r_wrist_px['x']) // 2, (l_wrist_px['y'] + r_wrist_px['y']) // 2
        rotation = 0.25 * pi - atan2(-y_diff, x_diff)
        norm_rot = rotation - 2 * pi * floor((rotation + pi) / (2 * pi))

        ext_amount = 50  # TODO
        ext_l_wrist = {
            'x': l_wrist_px['x'] + (ext_amount * sin(norm_rot)),
            'y': l_wrist_px['y'] + (ext_amount * cos(norm_rot))
        }
        ext_r_wrist = {
            'x': r_wrist_px['x'] + (ext_amount * sin(-norm_rot)),
            'y': r_wrist_px['y'] + (ext_amount * cos(-norm_rot))
        }

        sq_size = (dist((ext_l_wrist['x'], ext_l_wrist['y']), (ext_r_wrist['x'], ext_r_wrist['y'])) / 2) / (sin(pi / 4))
        rr = RotatedRect()
        rr.center.x, rr.center.y = center_x, center_y
        rr.size.width, rr.size.height = sq_size, sq_size
        rr.angle = degrees(norm_rot)

        ret_zones.append(rr)

        ###${_TRACE_INFO}node.warn(f"{script_name} updated hand zone: {zone_hand}")${_TRACE_INFO} # noqa
        # center_x, center_y = (b_left + b_right) / 2, (b_top + b_bottom) / 2
        # ret_zones.append(zone_from_center_size_norm(center_x, center_y, max(b_right - b_left, b_top - b_bottom)))
    return ret_zones


def normalize_results(results, detection_zone):
    # TODO REAL
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
    return []


previous_bodies = []


def centernet_postprocess_FAKE(body):
    start_kp = .2
    end_kp = .6
    seventeen_kps = list(map(lambda x: start_kp + (x / 19 * (end_kp - start_kp)), range(1, 18)))
    return [seventeen_kps], [seventeen_kps], [[start_kp, start_kp, end_kp, end_kp, .6]]


while True:
    nn_body_data: NNData = node.io['body_nn_data'].get()
    body = nn_body_data.getLayerFp16("output")
    node.warn(f"any found {body}")
    assert not any([score > .01 for score in body[55::56]])
    ###${_TRACE4} # noqa
    frame: ImgFrame = node.io['body_nn_frame'].get()
    ###${_TRACE4} # noqa
    ###${_TRACE_INFO}node.warn(f"{script_name} processing frame {frame.getSequenceNum()}")${_TRACE_INFO} # noqa

    # TODO REAL
    # bodies_x, bodies_y, body_boxes = centernet_postprocess(body)
    bodies_x, bodies_y, body_boxes = centernet_postprocess_FAKE(body)

    ###${_TRACE_INFO}node.warn(f"{script_name} got body keypoints")${_TRACE_INFO} # noqa

    # Calculate pre focus zones
    palm_zones = get_focus_zones(bodies_x, bodies_y, body_boxes)

    ###${_TRACE_INFO}node.warn(f"{script_name} focus zone calculation complete")${_TRACE_INFO} # noqa

    if any(palm_zones):
        previous_bodies = []
        for i, rr in enumerate(palm_zones):
            ###${_TRACE1}node.warn(f"Body pre focusing zone: ({x_min}, {y_min}), ({x_max}, {y_max})")${_TRACE1} # noqa

            cfg_pre_pd = ImageManipConfig()
            cfg_pre_pd.setCropRotatedRect(rr, False)
            cfg_pre_pd.setResize(128, 128)
            node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)

            nn_data: NNData = node.io['pd_data'].get()
            # TODO may need to set seqNumber from the frame above if the NN node doesn't
            # pd_detections = nn_data.getLayerFp16("result")
            palm_data = nn_data.getLayerFp16("result")
            normalized_palms = normalize_results(nn_data.getLayerFp16("result"), rr)
            passed_data = NNData(int(body.__sizeof__() + palm_data.__sizeof__() + normalized_palms.__sizeof__()))
            passed_data.setLayer("Identity", body)
            passed_data.setLayer("result", palm_data)
            if len(normalized_palms) > 0:
                nn_data.setLayer("palm_results_processed", normalized_palms)
            node.io['processed_pd'].send(nn_data)
    else:
        ###${_TRACE1}node.warn(f"Body pre focusing zone: None")${_TRACE1} # noqa
        node.io['early_out_pd'].send(nn_body_data)

    ###${_TRACE4} # noqa
    adjusted_boxes = []
    for box in body_boxes:
        ret = [
            int(((cord * img_h) + pad_h) if idx % 2 == 0 else cord * img_h)
            for (idx, cord)
            in enumerate(box[:4])
        ]
        ret.append(box[4])
        adjusted_boxes.append(ret)

    # px = (norm * height) - pad
    x_serial = marshal.dumps({
        "body_boxes": adjusted_boxes,
        "palm_search_zones": [
            {'center': (z.center.x, z.center.y), 'size': (z.size.width, z.size.height), 'angle': radians(z.angle)}
            for z in palm_zones
        ]
    })

    debug_message = Buffer(len(x_serial))
    debug_message.setData(x_serial)
    node.io['palm_trace4_output'].send(debug_message)
    ###${_TRACE4} # noqa
