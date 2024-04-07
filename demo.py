#!/usr/bin/env python3


from HandTrackerRenderer import HandTrackerRenderer
import argparse
import socket

def receive_messages(sock):
    while True:
        print("checking messages:")
        try:
            data = conn.recv(1024).decode()
            print("Received:", data)
            if not data:
                print("Connection closed by the client")
                break
            if data.startswith('interact'):
                # Split the data into interaction mode and file path
                parts = data.split(' ', 1)  # Only split on the first space
                if len(parts) == 2:
                    interaction_mode, interaction_file = parts
                    print("Received interaction mode:", interaction_mode)
            
                    if interaction_mode == 'interact2D':
                        renderer.interaction_mode = 'interact2D'
                        renderer.draw_mode = False
                        renderer.interact_2d = True
                        renderer.interact_3d = False                        
                        renderer.interaction_file = interaction_file
                        renderer.image_max = interaction_file
                        print("Received interaction file:", interaction_file)
                    elif interaction_mode == 'interact3D':
                        renderer.interaction_mode = 'interact3D'
                        renderer.draw_mode = False
                        renderer.interact_2d = False
                        renderer.interact_3d = True
                        renderer.interaction_file = interaction_file
                        renderer.model_path = interaction_file
                        print("Received interaction file:", interaction_file)
                else:
                    print("Error: Received data does not conform to expected format.")
            elif data == 'draw':
                print("Received draw mode")
                renderer.interaction_mode = "draw"
                renderer.draw_mode = True
            elif data == "hide":
                print("Received hide mode")
                renderer.hide_extras = True
                renderer.show_fps = False
            elif data == "show":
                print("Received show mode")
                renderer.hide_extras = False
                renderer.show_fps = True
            elif data == "none":
                print("Received none mode")
                renderer.interaction_mode = None
                renderer.interact_2d = False
                renderer.interact_3d = False
                renderer.draw_mode = False
            elif data.startswith("change_drawing_color"):
                print("Received Color Changing")
                color_values = data.split(" ")[1:]
                if len(color_values) == 3:
                    r, g, b = map(int, color_values)
                    renderer.line_color = (r, g, b)
                    print("Line Color", renderer.line_color)
            elif data.startswith("change_stl_color"):
                print("Received STL Color Changing")
                color_values = data.split(" ")[1:]
                if len(color_values) == 3:
                    r, g, b = map(int, color_values)
                    renderer.model_color = (b, g, r)
                    print("Model Color", renderer.model_color)
            elif data.startswith("change_lighting"):
                print("Received Lighting Changing")
                lighting_values = data.split(" ")[1:]
                if len(lighting_values) == 3:
                    x, y, z = map(float, lighting_values)
                    renderer.lighting = (x, y, z)
                    print("Lighting", renderer.lighting)
                
        except socket.error as e:
            print(f"Socket error: {e}")
            break 
            
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument('--no_lm', action="store_true", 
                    help="Only the palm detection model is run (no hand landmark model)")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Landmark model 'full', 'lite', 'sparse' or path to a blob file")
parser_tracker.add_argument('--use_world_landmarks', action="store_true", 
                    help="Fetch landmark 3D coordinates in meter")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Solo mode: detect one hand max. If not used, detect 2 hands max (Duo mode)")                    
parser_tracker.add_argument('-xyz', "--xyz", action="store_true", 
                    help="Enable spatial location measure of palm centers")
parser_tracker.add_argument('-g', '--gesture', action="store_true", 
                    help="Enable gesture recognition")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument("-r", "--resolution", choices=['full', 'ultra'], default='full',
                    help="Sensor resolution: 'full' (1920x1080) or 'ultra' (3840x2160) (default=%(default)s)")
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")   
parser_tracker.add_argument("-lh", "--use_last_handedness", action="store_true",
                    help="Use last inferred handedness. Otherwise use handedness average (more robust)")                            
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=10,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")
parser_tracker.add_argument('--dont_force_same_image', action="store_true",
                    help="(Edge Duo mode only) Don't force the use the same image when inferring the landmarks of the 2 hands (slower but skeleton less shifted)")
parser_tracker.add_argument('-lmt', '--lm_nb_threads', type=int, choices=[1,2], default=2, 
                    help="Number of the landmark model inference threads (default=%(default)i)")  
parser_tracker.add_argument('-t', '--trace', type=int, nargs="?", const=1, default=0, 
                    help="Print some debug infos. The type of info depends on the optional argument.")                
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-o', '--output', 
                    help="Path to output video file")
parser_renderer.add_argument('-d', '--draw', action="store_true", help="Enable drawing with index finger")
parser_renderer.add_argument('-sh', '--hide', action="store_true", help="Hide XYZ coordinates, hand skeletons, and gesture recognition text")
parser_renderer.add_argument(
    '--interact2D', action="store_true", help="Enable 2D object interaction")
parser_renderer.add_argument(
    '--interact3D', action="store_true", help="Enable 3D object interaction")
parser_renderer.add_argument("--fullscreen", action="store_true", help="Enable fullscreen mode")
parser_renderer.add_argument("--virtual_cam", action="store_true", help="Send frames to virtual camera instead of displaying in the video window")
parser_renderer.add_argument("--messages", action="store_true", help="Enable Sockets to Send Messages back and forth")
parser_renderer.add_argument("--interaction_mode", type=str,
                             help="Allow For Interaction Mode to be set to Drawing, 2D or 3D")
parser_renderer.add_argument("--interaction_file", type=str,
                             help="Allow For Interaction File to be set for 2D or 3D injection")

args = parser.parse_args()
dargs = vars(args)
# args.internal_frame_height = 600 if args.internal_frame_height is None else args.internal_frame_height
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

if args.edge:
    from HandTrackerEdge import HandTracker
    tracker_args['use_same_image'] = not args.dont_force_same_image
else:
    from HandTracker import HandTracker

if args.messages:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 12345))
    print("Waiting for a connection...")
    sock.listen(10)
    conn, addr = sock.accept()
    print(f"Connected by {addr}")

    
tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_world_landmarks=args.use_world_landmarks,
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        use_handedness_average=not args.use_last_handedness,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=args.lm_nb_threads,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output,
        draw_mode=args.draw,
        hide_extras=args.hide,
        interact_2d=args.interact2D,
        interaction_mode=args.interaction_mode,
        interaction_file=args.interaction_file,
        interact_3d=args.interact3D,
        fullscreen=args.fullscreen,
        virtual_cam=args.virtual_cam)

# Start a separate thread to receive messages from the controller
import threading
if args.messages:
    print("Starting message_thread")
    message_thread = threading.Thread(target=receive_messages, args=(conn,))
    message_thread.start()

while True:
    try:
        # Run hand tracker on next frame
        # 'bag' contains some information related to the frame 
        # and not related to a particular hand like body keypoints in Body Pre Focusing mode
        # Currently 'bag' contains meaningful information only when Body Pre Focusing is used
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Draw hands
        frame = renderer.draw(frame, hands, bag)
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
    except Exception as e:
        print("Error occurred", str(e))
        break

renderer.exit()
tracker.exit()
if(args.messages):
    conn.close()
    sock.close()