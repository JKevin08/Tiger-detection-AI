
import cv2
from ultralytics import YOLO

# -------- CONFIGURATION --------

MODEL = "/home/dant3/ecs-project-1/tiger-training/tiger-v1/weights/best.pt"

CAMERA_INDEX = 0
CONFIDENCE = 0.25

# --------------------------------

print("[+] Loading YOLO model...")
model = YOLO(MODEL)

print("[+] Opening camera...")
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError(
        f"Cannot open camera /dev/video{CAMERA_INDEX}"
    )

print("[+] Camera started.")
print("[+] Press Q to quit.")

try:
    while True:
        success, frame = camera.read()

        if not success:
            print("[!] Failed to read camera frame.")
            break

        # Run object detection
        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        # Display YOLO bounding boxes and labels
        display_frame = result.plot()

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                print(
                    f"Detected: {class_name} | "
                    f"Confidence: {confidence:.2%}"
                )
        else:
            print("No objects detected.")

        cv2.imshow("YOLO Detection Check", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("[+] Camera closed.")
