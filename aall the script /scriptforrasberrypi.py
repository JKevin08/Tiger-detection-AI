import os
import time
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO




MODEL_PATH = "fil the ncnn model file" # do it properly or its gonna be your worst experience

CONFIDENCE = 0.50
STOP_DELAY = 30                 # Seconds after last tiger detection
CAMERA_SIZE = (1280, 720)
OUTPUT_FPS = 30

PROJECT_DIR = Path.home() / "yolo"
SAVE_DIR = PROJECT_DIR / "recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


SHOW_PREVIEW = bool(
    os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
)




recording = False
video_process = None
current_filename = None
last_detection_time = None
preview_available = SHOW_PREVIEW
camera_started = False




if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "FFmpeg is not installed. Install it with:\n"
        "sudo apt install ffmpeg"
    )

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"NCNN model folder not found: {MODEL_PATH}"
    )




print("[+] Loading tiger detection model...")

model = YOLO(MODEL_PATH)
print("[+] Model classes:", model.names)

names = model.names

if isinstance(names, dict):
    class_items = names.items()
else:
    class_items = enumerate(names)

tiger_ids = [
    class_id
    for class_id, name in class_items
    if str(name).strip().lower() == "tiger"
]

if not tiger_ids:
    raise RuntimeError(
        "No class named 'tiger' was found in the model."
    )




def start_recording(frame):
    global recording, video_process, current_filename

    height, width = frame.shape[:2]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_filename = SAVE_DIR / f"tiger_{timestamp}.mp4"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

       
        "-use_wallclock_as_timestamps", "1",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(OUTPUT_FPS),
        "-i", "pipe:0",

        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "vfr",
        "-movflags", "+faststart",
        str(current_filename),
    ]

    print(f"\n[REC] Starting: {current_filename}")

    try:
        video_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
        )
        recording = True

    except (OSError, subprocess.SubprocessError) as error:
        video_process = None
        recording = False
        current_filename = None
        print(f"[!] Could not start FFmpeg: {error}")


def stop_recording():
    global recording, video_process
    global current_filename, last_detection_time

    process = video_process
    filename = current_filename

   
    video_process = None
    recording = False
    current_filename = None
    last_detection_time = None

    if process is None:
        return

    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
    except (BrokenPipeError, OSError):
        pass

    try:
        return_code = process.wait(timeout=15)

        if return_code == 0:
            print(f"[+] Video saved: {filename}")
        else:
            print(
                f"[!] FFmpeg exited with code {return_code}. "
                "The video may be incomplete."
            )

    except subprocess.TimeoutExpired:
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        print("[!] FFmpeg took too long to finish and was stopped.")




camera = None

try:
    print("[+] Opening Raspberry Pi camera...")

    camera = Picamera2()

    camera.configure(
        camera.create_video_configuration(
            main={
                "size": CAMERA_SIZE,
                "format": "RGB888",
            }
        )
    )

    camera.start()
    camera_started = True

   
    time.sleep(2)

    print("[+] Camera started.")
    print("[+] Waiting for a tiger...")

    if preview_available:
        print("[+] Press Q in the preview window to quit.")
    else:
        print("[+] No desktop preview available; press Ctrl+C to quit.")

    while True:
       
        frame = camera.capture_array()

        if frame is None or frame.size == 0:
            print("[!] Failed to read camera frame.")
            time.sleep(0.1)
            continue

        
        result = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False,
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
            elapsed = (
                now - last_detection_time
                if last_detection_time is not None
                else 0
            )

           
            if not tiger_detected and elapsed >= STOP_DELAY:
                stop_recording()

            if recording:
                if tiger_detected:
                    status = "TIGER DETECTED - RECORDING"
                else:
                    remaining = max(0, STOP_DELAY - elapsed)
                    status = f"NO TIGER - STOPPING IN {remaining:.0f}s"

                cv2.putText(
                    display_frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

               
                try:
                    if video_process is None or video_process.stdin is None:
                        raise BrokenPipeError("FFmpeg input is unavailable")

                    video_process.stdin.write(display_frame.tobytes())

                except (BrokenPipeError, OSError) as error:
                    print(f"\n[!] Video writing failed: {error}")
                    stop_recording()

        
        if preview_available:
            try:
                cv2.imshow("Tiger Detection", display_frame)

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("\n[+] Quit requested.")
                    break

            except cv2.error as error:
                print(f"[!] Preview unavailable: {error}")
                preview_available = False
                cv2.destroyAllWindows()
                print("[+] Continuing without preview. Press Ctrl+C to quit.")

except KeyboardInterrupt:
    print("\n[+] Stopping tiger detection...")

finally:
   
    if recording or video_process is not None:
        stop_recording()

    if camera is not None:
        if camera_started:
            camera.stop()

        camera.close()

    if preview_available:
        cv2.destroyAllWindows()

    print("[+] Camera closed.")
    print(f"[+] Recordings directory: {SAVE_DIR}")
    print("[+] Program ended.")
