
import os
import time
import math
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "/home/jude/yolo/models/best_ncnn_model"

CONFIDENCE = 0.50
STOP_DELAY = 30
CAMERA_SIZE = (1280, 720)
FPS = 30

SAVE_DIR = Path.home() / "yolo" / "recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


SHOW_PREVIEW = bool(
    os.environ.get("DISPLAY")
    or os.environ.get("WAYLAND_DISPLAY")
)



camera = None
camera_started = False

recording = False
video_process = None
current_filename = None

no_tiger_since = None
last_countdown_value = None




if shutil.which("ffmpeg") is None:
    raise RuntimeError(
        "FFmpeg is missing. Install it using:\n"
        "sudo apt install ffmpeg"
    )

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        f"NCNN model folder not found: {MODEL_PATH}"
    )




print("[+] Loading tiger detection model...", flush=True)

model = YOLO(MODEL_PATH)

print(f"[+] Model classes: {model.names}", flush=True)

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
        "No class named 'Tiger' found in the model."
    )




def start_recording(frame):
    global recording, video_process, current_filename

    height, width = frame.shape[:2]

    timestamp = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    current_filename = (
        SAVE_DIR / f"tiger_{timestamp}.mp4"
    )

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",

        #
        "-use_wallclock_as_timestamps", "1",
        "-f", "rawvideo",
        "-pixel_format", "bgr24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(FPS),
        "-i", "pipe:0",

       
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        "-fps_mode", "vfr",
        "-movflags", "+faststart",

        str(current_filename),
    ]

    print(
        f"[REC] Recording started: {current_filename}",
        flush=True
    )

    try:
        video_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
        )
        recording = True

    except OSError as error:
        video_process = None
        recording = False
        current_filename = None

        print(
            f"[ERROR] Could not start FFmpeg: {error}",
            flush=True
        )


def stop_recording():
    global recording, video_process, current_filename
    global no_tiger_since, last_countdown_value

    process = video_process
    filename = current_filename

    
    video_process = None
    recording = False
    current_filename = None
    no_tiger_since = None
    last_countdown_value = None

    if process is None:
        return

    try:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()

        return_code = process.wait(timeout=30)

        if return_code == 0 and filename and filename.exists():
            print("\n" + "=" * 48, flush=True)
            print("[+] VIDEO SAVED SUCCESSFULLY!", flush=True)
            print(f"[+] File: {filename}", flush=True)
            print(
                f"[+] Size: {filename.stat().st_size / 1024:.1f} KB",
                flush=True
            )
            print("=" * 48 + "\n", flush=True)

        else:
            print(
                f"[ERROR] FFmpeg exited with code {return_code}. "
                "Check whether the video was saved correctly.",
                flush=True
            )

    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
        print(
            f"[ERROR] Could not finalize the video cleanly: {error}",
            flush=True
        )

        if process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()




try:
    print("[+] Opening Raspberry Pi camera...", flush=True)

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

    print("[+] Camera started.", flush=True)
    print("[+] Waiting for a tiger...", flush=True)
    print(f"[+] Confidence threshold: {CONFIDENCE}", flush=True)
    print(f"[+] Stop delay: {STOP_DELAY} seconds", flush=True)
    print(f"[+] Save directory: {SAVE_DIR}", flush=True)
    print("[+] Press Ctrl+C to quit safely.", flush=True)

    while True:
        frame = camera.capture_array()

        if frame is None or frame.size == 0:
            print("[WARNING] Failed to read camera frame.", flush=True)
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
                if int(box.cls[0]) in tiger_ids:
                    tiger_detected = True
                    break

        now = time.monotonic()



        if tiger_detected:
            if not recording:
                print(
                    "\n[TIGER DETECTED] Tiger detected!",
                    flush=True
                )
                start_recording(frame)

            elif no_tiger_since is not None:
                print(
                    "\n[TIGER DETECTED AGAIN] "
                    "Countdown cancelled. Recording continues!",
                    flush=True
                )

            # Cancel the disappearance countdown.
            no_tiger_since = None
            last_countdown_value = None

        # ------------------------------------------
        # TIGER NO LONGER DETECTED
        # ------------------------------------------

        elif recording:
            if no_tiger_since is None:
                no_tiger_since = now
                last_countdown_value = None

                print(
                    "\n[TIGER GONE] Tiger no longer detected.",
                    flush=True
                )
                print(
                    f"[+] Starting {STOP_DELAY}-second countdown.",
                    flush=True
                )

            elapsed = now - no_tiger_since
            remaining = max(
                0,
                math.ceil(STOP_DELAY - elapsed)
            )

           
            if remaining != last_countdown_value:
                print(
                    f"[COUNTDOWN] {remaining} "
                    f"second{'s' if remaining != 1 else ''} remaining...",
                    flush=True
                )
                last_countdown_value = remaining

            # Finalize the file once the timer expires.
            if elapsed >= STOP_DELAY:
                print(
                    "\n[TIMER COMPLETE] Countdown finished.",
                    flush=True
                )
                stop_recording()


        display_frame = result.plot()

        if recording and video_process is not None:
            try:
                if video_process.stdin is None:
                    raise BrokenPipeError(
                        "FFmpeg input pipe is unavailable."
                    )

                video_process.stdin.write(
                    display_frame.tobytes()
                )

            except (BrokenPipeError, OSError) as error:
                print(
                    f"\n[ERROR] Video writing failed: {error}",
                    flush=True
                )
                stop_recording()

 

        if SHOW_PREVIEW:
            try:
                cv2.imshow(
                    "Tiger Detection",
                    display_frame
                )

                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("[+] Quit requested.", flush=True)
                    break

            except cv2.error:
                SHOW_PREVIEW = False
                cv2.destroyAllWindows()

                print(
                    "[+] Preview unavailable. "
                    "Terminal updates will continue.",
                    flush=True
                )

except KeyboardInterrupt:
    print("\n[+] Stop requested by user.", flush=True)

finally:
    #
    if recording or video_process is not None:
        print("[+] Finalizing active recording...", flush=True)
        stop_recording()

    if camera is not None:
        if camera_started:
            camera.stop()

        camera.close()

    cv2.destroyAllWindows()

    print("[+] Camera closed.", flush=True)
    print(f"[+] Recordings are in: {SAVE_DIR}", flush=True)
    print("[+] Program ended.", flush=True)
