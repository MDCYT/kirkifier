import sys
import os
import subprocess
import shutil
from pathlib import Path
from random import randint
from contextlib import contextmanager
from tqdm import tqdm
import cv2
import insightface
from insightface.app import FaceAnalysis
from cv2 import imread, imwrite

@contextmanager
def suppress_output():
   with open(os.devnull, 'w') as devnull:
       old_stdout, old_stderr = sys.stdout, sys.stderr
       sys.stdout, sys.stderr = devnull, devnull
       try:
           yield
       finally:
           sys.stdout, sys.stderr = old_stdout, old_stderr

def initialize_faceanalysis_and_swapper() -> tuple[FaceAnalysis, insightface.model_zoo.model_zoo.INSwapper]:
   faceanalysis: FaceAnalysis = FaceAnalysis(name="buffalo_l")
   faceanalysis.prepare(ctx_id=0, det_size=(640, 640))
   swapper: insightface.model_zoo.model_zoo.INSwapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
   return faceanalysis, swapper

def get_video_fps(video_path: str) -> float:
   cap = cv2.VideoCapture(video_path)
   fps = cap.get(cv2.CAP_PROP_FPS)
   cap.release()
   return fps

def extract_frames(video_path: str):
   """Extract all frames to unprocessed_frames/"""
   os.makedirs('unprocessed_frames', exist_ok=True)
   subprocess.run([
       'ffmpeg', '-i', video_path,
       '-q:v', '2',
       'unprocessed_frames/frame_%04d.png'
   ], check=True, capture_output=True)

def extract_audio(video_path: str) -> str:
   """Extract audio to audio.aac"""
   subprocess.run([
       'ffmpeg', '-i', video_path,
       '-map', '0:a', '-acodec', 'copy',
       'audio.aac'
   ], check=True, capture_output=True)
   return 'audio.aac'

def reconstruct_video(fps: float, audio_path: str, output_path: str):
   """Combine processed frames with audio"""
   os.makedirs('processed_frames', exist_ok=True)
   subprocess.run([
       'ffmpeg', '-framerate', str(fps),
       '-i', 'processed_frames/frame_%04d.png',
       '-i', audio_path,
       '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
       '-c:a', 'copy', '-shortest',
       output_path
   ], check=True, capture_output=True)

def cleanup(audio_path: str):
   """Remove intermediate files"""
   shutil.rmtree('unprocessed_frames', ignore_errors=True)
   shutil.rmtree('processed_frames', ignore_errors=True)
   if os.path.exists(audio_path):
       os.remove(audio_path)

def kirkify_frame(frame_path: str, output_path: str, faceanalysis: FaceAnalysis, swapper: insightface.model_zoo.model_zoo.INSwapper, kirk_face):
   """Process a single frame"""
   img = imread(frame_path)
   faces = faceanalysis.get(img)
   
   if faces:  # Only process if faces detected
       res = img.copy()
       for face in faces:
           res = swapper.get(res, face, kirk_face, paste_back=True)
       
       imwrite(output_path, res)
       return True
   else:
       # Copy unchanged if no faces
       imwrite(output_path, img)
       return False


def process_all_frames(faceanalysis: FaceAnalysis, swapper: insightface.model_zoo.model_zoo.INSwapper):
    """Process each frame in unprocessed_frames/ with tqdm progress bar"""
    os.makedirs('processed_frames', exist_ok=True)
    frame_files = [f for f in sorted(os.listdir('unprocessed_frames')) if f.endswith('.png')]
    kirk = imread(f'kirks/kirk_{randint(0, 2)}.jpg')
    kirk_face = faceanalysis.get(kirk)[0]
    # tqdm progress bar
    for filename in tqdm(frame_files, desc="Processing frames", unit="frame"):
        input_path = f'unprocessed_frames/{filename}'
        output_path = f'processed_frames/{filename}'
        kirkify_frame(input_path, output_path, faceanalysis, swapper, kirk_face)

def kirkify_video(TARGET_PATH, OUTPUT_PATH):
    print("Initializing models...")
    with suppress_output():
        FACE_ANALYSIS, FACE_SWAPPER = initialize_faceanalysis_and_swapper()

    print("Extracting frames...")
    extract_frames(TARGET_PATH)
    
    print("Extracting audio...")
    AUDIO_PATH = extract_audio(TARGET_PATH)
    
    print("Processing frames...")
    process_all_frames(FACE_ANALYSIS, FACE_SWAPPER)
    
    print("Reconstructing video...")
    FPS = get_video_fps(TARGET_PATH)
    reconstruct_video(FPS, AUDIO_PATH, OUTPUT_PATH)
    
    print("Cleaning up...")
    cleanup(AUDIO_PATH)
    
    print(f"Done! Output saved to {OUTPUT_PATH}")

def kirkify_image(TARGET_PATH, OUTPUT_PATH):
    print("Initializing models...")
    with suppress_output():
        FACE_ANALYSIS, FACE_SWAPPER = initialize_faceanalysis_and_swapper()

    kirk = imread(f'kirks/kirk_{randint(0, 2)}.jpg')
    kirk_face = FACE_ANALYSIS.get(kirk)[0]

    print("Kirkifying...")
    FACE_DETECTED = kirkify_frame(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, kirk_face)

    if not FACE_DETECTED:
        print("No faces detected. Image unchanged.")

    print(f"Done! Output saved to {OUTPUT_PATH}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        initialize_faceanalysis_and_swapper()
        print("Initialized!")
        exit()

    if len(sys.argv) < 2:
        print("Usage: python3 kirkify.py <input_media> [output_path]")
        exit(1)

    TARGET_PATH = sys.argv[1]

    if not Path(TARGET_PATH).exists():
        print("ERROR: target path not real")
        exit(1)
    
    IS_IMAGE = False
    FILE_EXT = os.path.splitext(TARGET_PATH)[1].lower()

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.flv'}

    if FILE_EXT in image_extensions:
        IS_IMAGE = True
    elif FILE_EXT in video_extensions:
        IS_IMAGE = False
    else:
        raise ValueError('Must be an image or video.')

    
    OUTPUT_PATH = sys.argv[2] if len(sys.argv) > 2 else f"output{FILE_EXT}"


    if IS_IMAGE:
        kirkify_image(TARGET_PATH, OUTPUT_PATH)
    else:
        kirkify_video(TARGET_PATH, OUTPUT_PATH)

if __name__ == "__main__":
   main()