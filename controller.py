import tkinter as tk
from tkinter import filedialog
import subprocess
import multiprocessing


class Controller:
    def __init__(self):
        self.interaction_mode = 'none'
        self.interaction_file = None
        self.process = None
        self.message_queue = multiprocessing.Queue()

        self.window = tk.Tk()
        self.window.title("Interaction Mode Controller")

        self.mode_var = tk.StringVar(value=self.interaction_mode)
        self.file_path_var = tk.StringVar()
        self.hide_var = tk.BooleanVar(value=False)
        self.draw_var = tk.BooleanVar(value=False)
        self.interact_2d_var = tk.BooleanVar(value=False)
        self.interact_3d_var = tk.BooleanVar(value=False)
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
            options_frame, text="Hide Extras", variable=self.hide_var)
        hide_checkbox.pack(side=tk.LEFT)

        draw_checkbox = tk.Checkbutton(
            options_frame, text="Draw", variable=self.draw_var)
        draw_checkbox.pack(side=tk.LEFT)

        interact_2d_checkbox = tk.Checkbutton(
            options_frame, text="Interact 2D", variable=self.interact_2d_var)
        interact_2d_checkbox.pack(side=tk.LEFT)

        interact_3d_checkbox = tk.Checkbutton(
            options_frame, text="Interact 3D", variable=self.interact_3d_var)
        interact_3d_checkbox.pack(side=tk.LEFT)

        virtual_cam_checkbox = tk.Checkbutton(
            options_frame, text="Virtual Camera", variable=self.virtual_cam_var)
        virtual_cam_checkbox.pack(side=tk.LEFT)

        fullscreen_checkbox = tk.Checkbutton(
            options_frame, text="Fullscreen", variable=self.fullscreen_var)
        fullscreen_checkbox.pack(side=tk.LEFT)

        start_button = tk.Button(
            self.window, text="Start Demo", command=self.start_demo)
        start_button.pack(pady=20)

    def set_interaction_mode(self, mode):
        if self.interaction_mode == 'draw' and mode in ['interact2D', 'interact3D']:
            tk.messagebox.showerror(
                "Error", "Cannot switch from Draw mode to Interact mode directly.")
            return
        elif self.interaction_mode in ['interact2D', 'interact3D'] and mode == 'draw':
            tk.messagebox.showerror(
                "Error", "Cannot switch from Interact mode to Draw mode directly.")
            return

        self.interaction_mode = mode
        self.mode_var.set(mode)

        if self.process:
            self.send_message(mode)
            if mode == 'interact2D' or mode == 'interact3D':
                if not self.interaction_file:
                    tk.messagebox.showerror(
                        "Error", "Please select an interaction file.")
                    return
                self.send_message(self.interaction_file)

    def select_file(self):
        if self.interaction_mode == 'interact2D':
            file_path = filedialog.askopenfilename(
                filetypes=[("PNG Files", "*.png")])
        elif self.interaction_mode == 'interact3D':
            file_path = filedialog.askopenfilename(
                filetypes=[("STL Files", "*.stl")])
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
            "--lm_model", ".\\models\\hand_landmark_lite-2022-11-12_sh4.blob"
        ]

        if self.interaction_mode != 'none':
            command.extend(["--interaction_mode", self.interaction_mode])
            if self.interaction_file:
                command.extend(["--interaction_file", self.interaction_file])
        else:
            if self.hide_var.get():
                command.append("--hide")
            if self.draw_var.get():
                command.append("--draw")
            if self.interact_2d_var.get():
                command.append("--interact2D")
            if self.interact_3d_var.get():
                command.append("--interact3D")

        if self.virtual_cam_var.get():
            command.append("--virtual_cam")

        if self.fullscreen_var.get():
            command.append("--fullscreen")

        self.process = subprocess.Popen(command)

    def send_message(self, message):
        self.message_queue.put(message)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    controller = Controller()
    controller.run()
