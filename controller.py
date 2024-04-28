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
        self.interact_button = None
        self.interaction_file_type = None
        self.file_extension = None
        self.create_widgets()
        self.update_interaction_button()
        self.drawing_submenu.pack_forget()
        self.color_submenu.pack_forget()


    def create_drag_drop_frame(self):
        drag_drop_frame = ttk.Frame(
            self.window, width=400, height=100, relief=tk.SOLID, borderwidth=2)
        drag_drop_frame.pack(pady=10)
        drag_drop_frame.pack_propagate(False)

        label = ttk.Label(drag_drop_frame, text="Drag and drop a file here")
        label.pack(expand=True)

        drag_drop_frame.drop_target_register(DND_FILES)
        drag_drop_frame.dnd_bind("<<Drop>>", self.drop_file)

        self.drag_drop_frame = drag_drop_frame


    def drop_file(self, event):
        file_path = event.data
        if '{' in file_path:
            # The file path is in the format "{path}"
            file_path = file_path[1:-1]  # Remove the curly braces
        _, self.file_extension = os.path.splitext(file_path)
        if self.file_extension.lower() in [".png", ".jpg", ".jpeg", ".stl", ".step", ".obj", ".glb"]:
            self.validate_and_set_file(file_path)
        else:
            messagebox.showerror(
                "Error", "Invalid file type. Please drop a PNG, JPEG, JPG, STL, STEP, OBJ, or GLB file.")
            
    def update_interaction_button(self):
        if self.interaction_file_type in ["interact2D", "interact3D"]:
            self.interact_button.config(state=tk.NORMAL)
        else:
            self.interact_button.config(state=tk.DISABLED)

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
            try:
                self.send_message(
                    f"change_drawing_color {color_map[color][0]} {color_map[color][1]} {color_map[color][2]}")
            except Exception as e:
                print("WARNING: (change_drawing_color) -" + str(e))

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

        self.interact_button = tk.Radiobutton(mode_frame, text="Interact", variable=self.mode_var, value='interact',
                                              command=lambda: self.set_interaction_mode('interact'))
        self.interact_button.pack(side=tk.LEFT)

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
        
        self.color_submenu = tk.Frame(self.window)
        tk.Label(self.color_submenu, text="Model Color:").pack(side=tk.LEFT)

        colors = ["Purple", "Red", "Green", "Blue", "Yellow", "Orange"]
        self.model_color_var = tk.StringVar(value="Purple")

        model_color_dropdown = tk.Menubutton(
            self.color_submenu,
            textvariable=self.model_color_var,
            width=10,
            bg=self.model_color_var.get().lower(),
            relief=tk.RAISED
        )
        model_color_dropdown.pack(side=tk.LEFT)

        model_color_menu = tk.Menu(model_color_dropdown, tearoff=0)
        for color in colors:
            model_color_menu.add_command(
                label=color,
                command=lambda c=color: [self.model_color_var.set(
                    c), self.change_stl_color(c), model_color_dropdown.config(bg=c.lower())]
            )
        model_color_dropdown["menu"] = model_color_menu

        self.lighting_submenu = tk.Frame(self.window)
        # self.lighting_submenu.pack(pady=10)

        tk.Label(self.lighting_submenu, text="Lighting:").pack(side=tk.LEFT)

        self.advanced_lighting_var = tk.BooleanVar(value=False)
        self.advanced_lighting_button = tk.Checkbutton(
            self.lighting_submenu,
            text="Advanced Lighting",
            variable=self.advanced_lighting_var,
            command=self.toggle_advanced_lighting
        )
        self.advanced_lighting_button.pack(side=tk.LEFT)

        self.preset_lighting_frame = tk.Frame(self.lighting_submenu)
        self.preset_lighting_frame.pack(side=tk.TOP)

        lighting_options = ["Low", "Medium", "High"]
        self.lighting_var = tk.StringVar(value="Low")

        for option in lighting_options:
            tk.Radiobutton(
                self.preset_lighting_frame,
                text=option,
                variable=self.lighting_var,
                value=option,
                command=lambda: self.change_lighting(self.lighting_var.get())
            ).pack(side=tk.LEFT)

        self.advanced_lighting_frame = tk.Frame(self.lighting_submenu)

        self.lighting_x_var = tk.DoubleVar(value=0.25)
        self.lighting_y_var = tk.DoubleVar(value=0.25)
        self.lighting_z_var = tk.DoubleVar(value=0.25)

        self.lighting_x_entry = None
        self.lighting_y_entry = None
        self.lighting_z_entry = None

        # Create sliders for x, y, and z lighting variables
        self.lighting_x_slider = tk.Scale(self.advanced_lighting_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                                        variable=self.lighting_x_var, command=self.update_lighting)
        self.lighting_x_slider.pack(side=tk.TOP, fill=tk.X)

        self.lighting_y_slider = tk.Scale(self.advanced_lighting_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                                        variable=self.lighting_y_var, command=self.update_lighting)
        self.lighting_y_slider.pack(side=tk.TOP, fill=tk.X)

        self.lighting_z_slider = tk.Scale(self.advanced_lighting_frame, from_=0, to=1, resolution=0.05, orient=tk.HORIZONTAL,
                                        variable=self.lighting_z_var, command=self.update_lighting)
        self.lighting_z_slider.pack(side=tk.TOP, fill=tk.X)

        # Create input text boxes for x, y, and z lighting variables
        self.lighting_x_entry = tk.Entry(self.advanced_lighting_frame, textvariable=self.lighting_x_var, validate="focusout",
                                        validatecommand=self.validate_lighting_entry)
        self.lighting_x_entry.pack(side=tk.TOP)

        self.lighting_y_entry = tk.Entry(self.advanced_lighting_frame, textvariable=self.lighting_y_var, validate="focusout",
                                        validatecommand=self.validate_lighting_entry)
        self.lighting_y_entry.pack(side=tk.TOP)

        self.lighting_z_entry = tk.Entry(self.advanced_lighting_frame, textvariable=self.lighting_z_var, validate="focusout",
                                        validatecommand=self.validate_lighting_entry)
        self.lighting_z_entry.pack(side=tk.TOP)

        self.advanced_lighting_frame.pack_forget()


    def toggle_advanced_lighting(self):
        if self.advanced_lighting_var.get():
            self.preset_lighting_frame.pack_forget()
            self.advanced_lighting_frame.pack(side=tk.TOP)
        else:
            self.advanced_lighting_frame.pack_forget()
            self.preset_lighting_frame.pack(side=tk.TOP)


    def change_lighting(self, lighting="Low"):
        if not self.advanced_lighting_var.get():
            lighting_map = {
                "Low": (0.25, 0.25, 0.25),
                "Medium": (0.5, 0.5, 0.5),
                "High": (0.75, 0.75, 0.75)
            }
            if lighting in lighting_map:
                try:
                    self.send_message(
                        f"change_lighting {lighting_map[lighting][0]} {lighting_map[lighting][1]} {lighting_map[lighting][2]}")
                except Exception as e:
                    print("WARNING: (change_lighting) - " + str(e))
        else:
            self.update_lighting()


    def update_lighting(self, *args):
        x = self.lighting_x_var.get()
        y = self.lighting_y_var.get()
        z = self.lighting_z_var.get()
        try:
            self.send_message(f"change_lighting {x} {y} {z}")
        except Exception as e:
            print("WARNING: (update_lighting) - " + str(e))
    
    def validate_lighting_entry(self):
        if self.lighting_x_entry is not None:
            try:
                value = float(self.lighting_x_entry.get())
                if 0 <= value <= 1:
                    self.lighting_x_var.set(value)
                else:
                    raise ValueError
            except ValueError:
                tk.messagebox.showwarning(
                    "Invalid Input", "Please enter a value between 0 and 1.")
                self.lighting_x_entry.delete(0, tk.END)
                self.lighting_x_entry.insert(0, self.lighting_x_var.get())

        if self.lighting_y_entry is not None:
            try:
                value = float(self.lighting_y_entry.get())
                if 0 <= value <= 1:
                    self.lighting_y_var.set(value)
                else:
                    raise ValueError
            except ValueError:
                tk.messagebox.showwarning(
                    "Invalid Input", "Please enter a value between 0 and 1.")
                self.lighting_y_entry.delete(0, tk.END)
                self.lighting_y_entry.insert(0, self.lighting_y_var.get())

        if self.lighting_z_entry is not None:
            try:
                value = float(self.lighting_z_entry.get())
                if 0 <= value <= 1:
                    self.lighting_z_var.set(value)
                else:
                    raise ValueError
            except ValueError:
                tk.messagebox.showwarning(
                    "Invalid Input", "Please enter a value between 0 and 1.")
                self.lighting_z_entry.delete(0, tk.END)
                self.lighting_z_entry.insert(0, self.lighting_z_var.get())
        return True

    def change_stl_color(self, color):
        color_map = {
            "Purple": (255, 0, 255),
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "Yellow": (255, 255, 0),
            "Orange": (255, 165, 0)
        }
        if color in color_map:
            try:
                self.send_message(
                    f"change_stl_color {color_map[color][0]} {color_map[color][1]} {color_map[color][2]}")
            except Exception as e:
                print("WARNING: (change_stl_color) - "+str(e))
            
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
            self.color_submenu.pack_forget()
            self.lighting_submenu.pack_forget()
        elif mode == "interact" and self.interaction_file_type == "interact3D":
            self.drawing_submenu.pack_forget()
            if self.file_extension.lower() in [".step", ".obj"]:
                self.color_submenu.pack_forget()
            else:
                self.color_submenu.pack(pady=10)
            self.lighting_submenu.pack(pady=10)
        else:
            self.drawing_submenu.pack_forget()
            self.color_submenu.pack_forget()
            self.lighting_submenu.pack_forget()

        if mode == 'interact' and self.interaction_file_type not in ['interact2D', 'interact3D']:
            tk.messagebox.showwarning(
                "Warning", "Please select a valid file (PNG, JPG, JPEG, or STL) before selecting Interact mode.")
            return

        self.interaction_mode = mode
        self.mode_var.set(mode)

        if self.process:
            if mode == 'interact':
                interaction_mode = 'interact2D' if self.interaction_file_type == 'interact2D' else 'interact3D'
                self.send_message(interaction_mode + " " +
                                  self.interaction_file)
            else:
                self.send_message(mode)

    def validate_and_set_file(self, file_path):
        if file_path and os.path.exists(file_path):
            _, self.file_extension = os.path.splitext(file_path)
            if self.file_extension.lower() in [".png", ".jpg", ".jpeg"]:
                self.interaction_file_type = "interact2D"
            elif self.file_extension.lower() in [".stl", ".step", ".obj", ".glb"]:
                self.interaction_file_type = "interact3D"
            else:
                self.interaction_file_type = None

            self.interaction_file = file_path
            self.file_path_var.set(file_path)
            self.update_interaction_button()

            if self.interaction_file_type in ['interact2D', 'interact3D']:
                # Automatically select the "interact" radio button
                self.set_interaction_mode('interact')
            else:
                # Reset to "none" mode if an invalid file is loaded
                self.set_interaction_mode('none')

            if self.process:
                interaction_mode = 'interact2D' if self.interaction_file_type == 'interact2D' else 'interact3D'
                self.send_message(interaction_mode + ' ' + file_path)
            if self.interaction_file_type == "interact3D":
                if self.file_extension.lower() in [".step", ".obj"]:
                    self.color_submenu.pack_forget()
                else:
                    self.color_submenu.pack(pady=10)
        else:
            messagebox.showerror(
                "Error", "File does not exist. Please select a valid file.")
            self.interaction_file = None
            self.interaction_file_type = None
            self.file_path_var.set('')
            self.update_interaction_button()

    def select_file(self):
        file_types = [("All Files", "*.*")]
        file_path = filedialog.askopenfilename(
            filetypes=file_types, title="Select File")
        self.validate_and_set_file(file_path)

    def start_demo(self):
        demo_thread = threading.Thread(target=self.run_demo)
        demo_thread.start()

    def run_demo(self):
        while True:
            try:
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
                    interaction_mode = 'interact2D' if self.interaction_file_type == 'interact2D' else 'interact3D'
                    command.extend(["--interaction_mode", interaction_mode])
                    if self.interaction_file:
                        command.extend(["--interaction_file", self.interaction_file])
                
                if self.interaction_mode == 'draw':
                    command.append("--draw")
                    
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
                    break

                time.sleep(1)
                if self.interaction_mode == "draw":
                    self.drawing_submenu.pack(pady=10)
                    time.sleep(1)
                    self.change_drawing_color(self.color_var.get())
                    time.sleep(0.25)
                elif self.interaction_mode == "interact" and self.interaction_file_type == "interact3D":
                    if self.file_extension.lower() in [".step", ".obj"]:
                        self.color_submenu.pack_forget()
                    else:
                        self.color_submenu.pack(pady=10)
                        time.sleep(1.5)
                        self.change_stl_color(self.model_color_var.get())
                    self.lighting_submenu.pack(pady=10)
                    time.sleep(1.5)
                    self.change_lighting()
                else:
                    self.drawing_submenu.pack_forget()
                    self.color_submenu.pack_forget()
                    self.lighting_submenu.pack_forget()
                print("Connection Established!")

                # Monitor the process
                while True:
                    if self.process.poll() is not None:
                        break
                    time.sleep(2)

                # Process has stopped, perform cleanup and restart
                print("Demo process stopped. Restarting...")
                self.stop_demo()

                print("Restarting demo...")

            except Exception as e:
                print("Error occurred in demo.py:", str(e))
                # Perform cleanup and restart
                self.stop_demo()
                time.sleep(1)

                # Check if the demo should be restarted
                if self.process is None:
                    break

                print("Restarting demo...")
        # Demo has been stopped, perform final cleanup
        self.stop_demo()   

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
        time.sleep(0.25)  # Add a small delay to ensure the process is terminated

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
