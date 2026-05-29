"""
Live load cell stream — auto-tares on startup, force axis fixed 0–2 N.
Type 't' + Enter to re-tare. Ctrl+C to quit.
"""

import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import time
import sys

PORT   = '/dev/tty.usbmodemF412FA6357AC2'
BAUD   = 115200
WINDOW = 200

times   = deque(maxlen=WINDOW)
weights = deque(maxlen=WINDOW)
forces  = deque(maxlen=WINDOW)
latest  = {"weight": 0.0, "force": 0.0}

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Error opening {PORT}: {e}")
    sys.exit(1)

# Auto-tare: wait for Arduino to boot then send tare command
time.sleep(2.0)
ser.write(b't')
print(f"Connected to {PORT}. Auto-tared. Type 't' + Enter to re-tare. Ctrl+C to quit.")

def read_serial():
    t0 = None
    while True:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line or ',' not in line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            ts_us, raw, weight, force = parts
            weight = float(weight)
            force  = float(force)
            ts_s   = int(ts_us) / 1e6
            if t0 is None:
                t0 = ts_s
            times.append(ts_s - t0)
            weights.append(weight)
            forces.append(force)
            latest["weight"] = weight
            latest["force"]  = force
        except (ValueError, serial.SerialException):
            continue

def tare_listener():
    while True:
        try:
            if input().strip().lower() == 't':
                ser.write(b't')
                print("Tare sent.")
        except (EOFError, KeyboardInterrupt):
            break

threading.Thread(target=read_serial,  daemon=True).start()
threading.Thread(target=tare_listener, daemon=True).start()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle("Live Load Cell Data")

line_w, = ax1.plot([], [], color='steelblue', lw=1.5)
line_f, = ax2.plot([], [], color='tomato',    lw=1.5)

ax1.set_ylabel("Weight (g)")
ax2.set_ylabel("Force (N)")
ax2.set_xlabel("Time (s)")
ax2.set_ylim(0, 2)
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)

title1 = ax1.set_title("")
title2 = ax2.set_title("")

def update(_):
    if not times:
        return line_w, line_f
    t = list(times)
    line_w.set_data(t, list(weights))
    line_f.set_data(t, list(forces))
    ax1.relim(); ax1.autoscale_view()
    ax2.set_xlim(t[0], max(t[-1], 1))
    title1.set_text(f"Weight: {latest['weight']:.2f} g")
    title2.set_text(f"Force:  {latest['force']:.4f} N")
    return line_w, line_f

ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

try:
    plt.tight_layout()
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    ser.close()
    print("Done.")
