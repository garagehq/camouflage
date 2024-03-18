import tkinter as tk
from tkinter import filedialog
import subprocess
import socket
import signal
import time

class Controller:
    def __init__(self):
        self.interaction_mode = 'none'
        self.interaction_file = None
        self.process = None
        self.socket = None
        self.window = tk.Tk()
        self.window.title("Interaction Mode Controller")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

        self.mode_var = tk.StringVar(value=self.interaction_mode)
        self.file_path_var = tk.StringVar()
        self.hide_var = tk.BooleanVar(value=False)
        self.virtual_cam_var = tk.BooleanVar(value=False)
        self.fullscreen_var = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        mode_frame = tk.Frame(self.window)
        mode_frame.pack(pady=10)

        tk.Label(mode_frame, text="Interaction Mode:").pack(side=tk.LEFT)

        none_button = tk.Radiobutton(mode_frame, text="None", variable=self.mode_var, value='none',
                                     command=lambda: self.set_interaction_mode('none'))
        none_button.pack(side=tk.LEFT)

        draw_button = tk.Radiobutton(mode_frame, text="Draw", variable=self.mode_var, value='draw',
                                     command=lambda: self.set_interaction_mode('draw'))
        draw_button.pack(side=tk.LEFT)

        interact2d_button = tk.Radiobutton(mode_frame, text="Interact 2D", variable=self.mode_var, value='interact2D',
                                           command=lambda: self.set_interaction_mode('interact2D'))
        interact2d_button.pack(side=tk.LEFT)

        interact3d_button = tk.Radiobutton(mode_frame, text="Interact 3D", variable=self.mode_var, value='interact3D',
                                           command=lambda: self.set_interaction_mode('interact3D'))
        interact3d_button.pack(side=tk.LEFT)

        file_frame = tk.Frame(self.window)
        file_frame.pack(pady=10)

        tk.Label(file_frame, text="Interaction File:").pack(side=tk.LEFT)

        file_path_entry = tk.Entry(
            file_frame, textvariable=self.file_path_var, width=40)
        file_path_entry.pack(side=tk.LEFT)

        browse_button = tk.Button(
            file_frame, text="Browse", command=self.select_file)
        browse_button.pack(side=tk.LEFT)

        options_frame = tk.Frame(self.window)
        options_frame.pack(pady=10)

        hide_checkbox = tk.Checkbutton(
            options_frame, text="Hide Extras", variable=self.hide_var, command=self.toggle_hide)
        hide_checkbox.pack(side=tk.LEFT)

        virtual_cam_checkbox = tk.Checkbutton(
            options_frame, text="Virtual Camera", variable=self.virtual_cam_var, command=self.toggle_virtual_cam)
        virtual_cam_checkbox.pack(side=tk.LEFT)

        fullscreen_checkbox = tk.Checkbutton(
            options_frame, text="Fullscreen", variable=self.fullscreen_var, command=self.toggle_fullscreen)
        fullscreen_checkbox.pack(side=tk.LEFT)

        self.start_button = tk.Button(
                self.window, text="START", command=self.toggle_demo)
        self.start_button.pack(pady=20)
        
    def toggle_demo(self):
        if self.process is None:
            self.start_demo()
            self.start_button.config(text="STOP")
        else:
            self.stop_demo()
            self.start_button.config(text="START")
    
    def on_close(self):
            self.stop_demo()
            self.window.quit()
    
    def stop_demo(self):
        if self.process:
            self.process.terminate()
            self.process = None
        if self.socket:
            self.socket.close()
            self.socket = None


    def set_interaction_mode(self, mode):
        if mode in ['interact2D', 'interact3D'] and not self.interaction_file:
            tk.messagebox.showerror(
                "Error", "Please select a valid interaction file before selecting Interact 2D or Interact 3D mode.")
            return

        self.interaction_mode = mode
        self.mode_var.set(mode)

        if self.process:
            self.send_message(mode)
            if mode in ['interact2D', 'interact3D']:
                self.send_message(self.interaction_file)

    def select_file(self):
        if self.interaction_mode == 'interact2D':
            file_path = filedialog.askopenfilename(
                filetypes=[("PNG Files", "*.png")],
                defaultextension=".png",
                title="Select PNG File"
            )
        elif self.interaction_mode == 'interact3D':
            file_path = filedialog.askopenfilename(
                filetypes=[("STL Files", "*.stl")],
                defaultextension=".stl",
                title="Select STL File"
            )
        else:
            file_path = None

        if file_path:
            self.interaction_file = file_path
            self.file_path_var.set(file_path)

    def start_demo(self):
        command = [
            "python", "demo.py",
            "--pd_model", ".\\models\\palm_detection_lite-2022-08-30_sh4.blob",
            "--gesture",
            "--lm_model", ".\\models\\hand_landmark_lite-2022-11-12_sh4.blob",
            "--messages"
        ]

        if self.interaction_mode != 'none':
            command.extend(["--interaction_mode", self.interaction_mode])
            if self.interaction_file:
                command.extend(["--interaction_file", self.interaction_file])

        if self.hide_var.get():
            command.append("--hide")
   
        if self.virtual_cam_var.get():
            command.append("--virtual_cam")

        if self.fullscreen_var.get():
            command.append("--fullscreen")

        self.process = subprocess.Popen(command)
        time.sleep(2)  # Adjust the delay as needed
        
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 12345))

    def send_message(self, message):
        self.socket.send(message.encode())

    def toggle_hide(self):
        if self.process:
            self.send_message('hide' if self.hide_var.get() else 'show')

    def toggle_virtual_cam(self):
        if self.process:
            self.process.terminate()
            self.start_demo()

    def toggle_fullscreen(self):
        if self.process:
            self.process.terminate()
            self.start_demo()

    def run(self):
        self.window.mainloop()
        if self.socket:
            self.socket.close()

def signal_handler(sig, frame):
    controller.on_close()
    
if __name__ == "__main__":
    controller = Controller()
    signal.signal(signal.SIGINT, signal_handler)
    controller.run()
