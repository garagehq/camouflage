"""
USB OAK-D Attach Script for WSL2

Attaches Luxonis/DepthAI devices to WSL with auto-attach enabled.
Run this script and keep it running while using the camera.

Usage:
    python usb_oakd.py
"""

import subprocess
import sys
import os

# Hardware IDs (VID:PID) for Luxonis devices
LUXONIS_HARDWARE_IDS = [
    '03e7:f63b',  # Myriad VPU / Luxonis Device (running mode)
    '03e7:f63c',  # Luxonis Bootloader
]


def main():
    print("=== USB OAK-D Auto-Attach for WSL2 ===\n")
    print("Attaching Luxonis devices to WSL with auto-attach...")
    print("Keep this running while using the camera. Press Ctrl+C to stop.\n")

    processes = []
    for hwid in LUXONIS_HARDWARE_IDS:
        cmd = f'usbipd attach --wsl --auto-attach --hardware-id {hwid}'
        print(f"  {cmd}")
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        processes.append((hwid, proc))

    print("\nAuto-attach running. Devices will re-attach automatically if disconnected.")
    print("Press Ctrl+C to stop.\n")

    try:
        import time
        while True:
            for hwid, proc in list(processes):
                ret = proc.poll()
                if ret is not None:
                    stdout, stderr = proc.communicate()
                    output = (stdout.decode() + stderr.decode()).strip()
                    if output and 'error' in output.lower():
                        print(f"[{hwid}] {output}")
                    # Restart the process if it exited
                    cmd = f'usbipd attach --wsl --auto-attach --hardware-id {hwid}'
                    new_proc = subprocess.Popen(
                        cmd,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                    )
                    processes.remove((hwid, proc))
                    processes.append((hwid, new_proc))
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nStopping auto-attach...")
        for hwid, proc in processes:
            proc.terminate()
        print("Done.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
