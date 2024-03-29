import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import socket
import signal
import time
import os
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading

class Controller:
    def __init__(self):
        self.interaction_mode = 'none'
        self.interaction_file = None
        self.process = None
        self.socket = None
        self.window = TkinterDnD.Tk()
        self.window.title("Interaction Mode Controller")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.mode_var = tk.StringVar(value=self.interaction_mode)
        self.file_path_var = tk.StringVar()
        self.hide_var = tk.BooleanVar(value=True)
        self.virtual_cam_var = tk.BooleanVar(value=False)
        self.fullscreen_var = tk.BooleanVar(value=False)
        self.interact2d_button = None
        self.interact3d_button = None
        self.interaction_file_type = None
        self.create_widgets()
        self.update_interaction_buttons()
        self.drawing_submenu.pack_forget()
    
    def create_drag_drop_frame(self):
        drag_drop_frame = ttk.Frame(self.window, width=400, height=100, relief=tk.SOLID, borderwidth=2)
        drag_drop_frame.pack(pady=10)
        drag_drop_frame.pack_propagate(False)

        label = ttk.Label(drag_drop_frame, text="Drag and drop a file here")
        label.pack(expand=True)

        drag_drop_frame.drop_target_register(DND_FILES)
        drag_drop_frame.dnd_bind("<<Drop>>", self.drop_file)

        self.drag_drop_frame = drag_drop_frame
    
    def drop_file(self, event):
        file_path = event.data
        self.validate_and_set_file(file_path)

    def update_interaction_buttons(self):
        if self.interaction_file_type == "interact2D":
            self.interact2d_button.config(state=tk.NORMAL)
            self.interact3d_button.config(state=tk.DISABLED)
        elif self.interaction_file_type == "interact3D":
            self.interact2d_button.config(state=tk.DISABLED)
            self.interact3d_button.config(state=tk.NORMAL)
        else:
            self.interact2d_button.config(state=tk.DISABLED)
            self.interact3d_button.config(state=tk.DISABLED)

    def change_drawing_color(self, color):
        color_map = {
            "Red": (0, 0, 255),
            "Green": (0, 255, 0),
            "Blue": (255, 0, 0),
            "Yellow": (0, 255, 255),
            "Purple": (255, 0, 255),
            "Orange": (0, 165, 255)
        }
        if color in color_map:
            self.send_message(
                f"change_drawing_color {color_map[color][0]} {color_map[color][1]} {color_map[color][2]}")
            
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

        self.interact2d_button = tk.Radiobutton(mode_frame, text="Interact 2D", variable=self.mode_var, value='interact2D',
                                                command=lambda: self.set_interaction_mode('interact2D'))
        self.interact2d_button.pack(side=tk.LEFT)
        
        self.interact3d_button = tk.Radiobutton(mode_frame, text="Interact 3D", variable=self.mode_var, value='interact3D',
                                                command=lambda: self.set_interaction_mode('interact3D'))
        self.interact3d_button.pack(side=tk.LEFT)

        self.drawing_submenu = tk.Frame(self.window)
        self.drawing_submenu.pack(pady=10)

        
        tk.Label(self.drawing_submenu, text="Drawing Color:").pack(side=tk.LEFT)

        colors = ["Green", "Red", "Blue", "Yellow", "Purple", "Orange"]
        self.color_var = tk.StringVar(value="Green")

        color_dropdown = tk.Menubutton(
            self.drawing_submenu,
            textvariable=self.color_var,
            width=10,
            bg=self.color_var.get().lower(),
            relief=tk.RAISED
        )
        color_dropdown.pack(side=tk.LEFT)

        color_menu = tk.Menu(color_dropdown, tearoff=0)
        for color in colors:
            color_menu.add_command(
                label=color,
                command=lambda c=color: [self.color_var.set(
                    c), self.change_drawing_color(c), color_dropdown.config(bg=c.lower())]
            )
        color_dropdown["menu"] = color_menu
        
        file_frame = tk.Frame(self.window)
        file_frame.pack(pady=10)

        tk.Label(file_frame, text="Interaction File:").pack(side=tk.LEFT)

        file_path_entry = tk.Entry(
            file_frame, textvariable=self.file_path_var, width=40)
        file_path_entry.pack(side=tk.LEFT)

        browse_button = tk.Button(
            file_frame, text="Browse", command=self.select_file)
        browse_button.pack(side=tk.LEFT)
        self.create_drag_drop_frame()

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
        self.drawing_submenu.pack_forget()


    def set_interaction_mode(self, mode):
        if mode == "draw" and self.process is not None:
            self.drawing_submenu.pack(pady=10)
        else:
            self.drawing_submenu.pack_forget()
        if mode == 'interact2D' and self.interaction_file_type != 'interact2D':
                tk.messagebox.showwarning("Warning", "Please select a valid PNG file before selecting Interact 2D mode.")
                return
        elif mode == 'interact3D' and self.interaction_file_type != 'interact3D':
            tk.messagebox.showwarning("Warning", "Please select a valid STL file before selecting Interact 3D mode.")
            return

        self.interaction_mode = mode
        self.mode_var.set(mode)
        
        if self.process:
            if mode.startswith('interact'):
                self.send_message(mode + " " + self.interaction_file)
            else:
                self.send_message(mode)


    def validate_and_set_file(self, file_path):
        if file_path and os.path.exists(file_path):
            _, file_extension = os.path.splitext(file_path)
            if file_extension.lower() in [".png", ".jpg", ".jpeg"]:
                self.interaction_file_type = "interact2D"
            elif file_extension.lower() == ".stl":
                self.interaction_file_type = "interact3D"
            else:
                self.interaction_file_type = None

            self.interaction_file = file_path
            self.file_path_var.set(file_path)
            self.update_interaction_buttons()

            if self.process and self.interaction_mode == 'interact3D':
                self.send_message('interact3D ' + file_path)
            if self.process and self.interaction_mode == 'interact2D':
                self.send_message('interact2D ' + file_path)
        else:
            messagebox.showerror(
                "Error", "File does not exist. Please select a valid file.")
            self.interaction_file = None
            self.interaction_file_type = None
            self.file_path_var.set('')
            self.update_interaction_buttons()

    def select_file(self):
        file_types = {'interact2D': (
            "Image Files", "*.png;*.jpg;*.jpeg"), 'interact3D': ("STL Files", "*.stl")}
        file_type = file_types.get(self.interaction_mode, ("All Files", "*.*"))
        file_path = filedialog.askopenfilename(filetypes=[
                                               file_type], defaultextension=file_type[1].split(".")[-1], title="Select File")
        self.validate_and_set_file(file_path)

    def start_demo(self):
        command = [
            "python", "demo.py",
            # "--pd_model", ".\\models\\palm_detection_lite-2022-08-30_sh4.blob",
            "--gesture",
            "--lm_model", ".\\models\\hand_landmark_lite-2022-11-12_sh4.blob",
            "--messages",
            "--edge",
            "-f 15"
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
        
        max_retries = 3
        retry_delay = 2
        
        print("Attempting the connection...")
        for retry in range(max_retries):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect(('localhost', 12345))
                break
            except ConnectionRefusedError as e:
                print(
                    f"Connection refused. Retrying in {retry_delay} seconds... (Attempt {retry + 1}/{max_retries})")
                time.sleep(retry_delay)
        else:
            print("Failed to establish socket connection after maximum retries.")
            self.stop_demo()
        
        if self.interaction_mode == "draw":
            self.drawing_submenu.pack(pady=10)
            time.sleep(1)
            self.change_drawing_color(self.color_var.get())
            
        else:
            self.drawing_submenu.pack_forget()
        print("Connection Established!")


    def display_process_output(self):
        while self.process.poll() is None:
            output = self.process.stdout.readline()
            if output:
                print(output.strip())

    def stop_demo(self):
        if self.process:
            self.process.terminate()
            self.process = None
        if self.socket:
            self.socket.close()
            self.socket = None
        self.drawing_submenu.pack_forget()
        time.sleep(1)  # Add a small delay before restarting the process

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
