
import cv2
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO



MODEL = "put your pytorch file path gng pls dont suffer"

CAMERA_INDEX = 0
CONFIDENCE = 0.50
STOP_DELAY = 30 

SAVE_DIR = Path("recordings")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

FPS = 30



if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "FFmpeg is not installed. Run: sudo pacman -S ffmpeg"
    )

print("[+] Loading tiger detection model...")
model = YOLO(MODEL)

print("[+] Model classes:", model.names)


tiger_ids = [
    class_id
    for class_id, name in model.names.items()
    if name.lower() == "tiger"
]

if not tiger_ids:
    raise RuntimeError(
        "No class named 'tiger' was found in the model. "
        "Check the class names printed above."
    )

print("[+] Opening camera...")
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not camera.isOpened():
    raise RuntimeError(
        f"Cannot open camera /dev/video{CAMERA_INDEX}. "
        "Try CAMERA_INDEX = 1."
    )

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



recording = False
video_process = None
last_detection_time = None
current_filename = None

print("[+] Camera started.")
print("[+] Waiting for a tiger...")
print("[+] Press Q to quit.")



def start_recording(frame):
    global video_process, recording, current_filename

    height, width = frame.shape[:2]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_filename = SAVE_DIR / f"tiger_{timestamp}.mp4"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(FPS),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(current_filename)
    ]

    video_process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE
    )

    recording = True
    print(f"\n[REC] Started: {current_filename}")


def stop_recording():
    global video_process, recording
    global last_detection_time, current_filename

    if video_process is not None:
        try:
            if video_process.stdin:
                video_process.stdin.close()

            return_code = video_process.wait()

            if return_code == 0:
                print(f"[+] Video saved: {current_filename}")
            else:
                print(
                    f"[!] FFmpeg exited with code {return_code}. "
                    "Check the terminal for errors."
                )

        except (BrokenPipeError, OSError) as error:
            print(f"[!] Error finalizing video: {error}")

        finally:
            video_process = None

    recording = False
    last_detection_time = None
    current_filename = None




try:
    while True:
        success, frame = camera.read()

        if not success:
            print("[!] Failed to read camera frame.")
            break

        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False
        )[0]

        tiger_detected = False

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0])

                if class_id in tiger_ids:
                    tiger_detected = True
                    break

        now = time.monotonic()

        if tiger_detected:
            last_detection_time = now

            if not recording:
                start_recording(frame)

        display_frame = result.plot()

        if recording:
            if last_detection_time is not None:
                elapsed = now - last_detection_time

                if not tiger_detected and elapsed >= STOP_DELAY:
                    stop_recording()

            if recording:
                if tiger_detected:
                    status = "TIGER DETECTED - RECORDING"
                else:
                    remaining = max(
                        0, STOP_DELAY - (now - last_detection_time)
                    )
                    status = f"NO TIGER - STOPPING IN {remaining:.0f}s"

                cv2.putText(
                    display_frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

              
                try:
                    video_process.stdin.write(
                        display_frame.tobytes()
                    )
                except (BrokenPipeError, OSError) as error:
                    print(f"[!] FFmpeg recording error: {error}")
                    stop_recording()

        else:
            status = (
                "TIGER DETECTED"
                if tiger_detected
                else "WAITING FOR TIGER"
            )

            cv2.putText(
                display_frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Tiger Detection Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
  
    if recording:
        stop_recording()

    camera.release()
    cv2.destroyAllWindows()

    print("[+] Camera closed.")
    print("[+] Program ended.")
