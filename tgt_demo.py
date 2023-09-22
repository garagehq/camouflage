#!/usr/bin/env python3

from TGTTrackerBpfEdge import TGTTracker
from HandTrackerRenderer import HandTrackerRenderer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser_tracker = parser.add_argument_group("Tracker arguments")
    parser_tracker.add_argument('-i', '--input', type=str, default='oak_d_sr_poe',
                                help="Path to video or image file to use as input (if not specified, use oak_d_sr_poe camera)")
    parser_tracker.add_argument('--no_lm', action="store_true",
                                help="Only the palm detection model is run (no hand landmark model)")
    parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                                help="(Duo mode only) Number of frames after only one hand is detected before calling "
                                     "palm detection (default=%(default)s)")
    # noinspection DuplicatedCode
    parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0,
                                help="Print some debug infos. The type of info depends on the optional argument.")
    parser_renderer = parser.add_argument_group("Renderer arguments")
    parser_renderer.add_argument('-o', '--output',
                                 help="Path to output video file")
    parser_tracker.add_argument('--no-laconic', action="store_true",
                                help="Only the palm detection model is run (no hand landmark model)")
    args = parser.parse_args()
    dargs = vars(args)

    tracker = TGTTracker(
        input_src=args.input,
        use_lm=not args.no_lm,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        trace=args.trace,
        laconic=not args.no_laconic
    )

    # noinspection DuplicatedCode
    renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

    while True:
        # Run hand tracker on next frame
        # 'bag' contains some information related to the frame
        # and not related to a particular hand like body keypoints in Body Pre Focusing mode
        # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
        frame, hands, bag = tracker.next_frame()
        if len(hands) > 0:
            print("hand")
        if frame is None: break
        # Draw hands
        frame = renderer.draw(frame, hands, bag)
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
    renderer.exit()
    tracker.exit()
