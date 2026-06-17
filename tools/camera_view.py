import cv2
import sys

def _find_cam(target_w=640, target_h=480, fallback=0):
    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w == target_w and h == target_h:
                return i
    return fallback

idx = _find_cam()
print(f"Opening camera {idx}...", flush=True)

cap = cv2.VideoCapture(idx)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("ERROR: could not open camera", flush=True)
    sys.exit(1)

print("Camera opened. Press Q or ESC to quit.", flush=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed", flush=True)
        continue
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.", flush=True)
