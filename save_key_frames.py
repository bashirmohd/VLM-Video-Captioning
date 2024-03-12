import os
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Uniform frame extraction")
    parser.add_argument("--video_input_dir_path", required=True,
                        help="Directory path containing videos for frame extraction.")
    parser.add_argument("--frames_dir_path", required=True, help="Output directory path for the extracted frames.")
    args = parser.parse_args()

    return args


def extract_uniform_frames(video_path, output_path, no_of_frames_to_extract):
    # Use OpenCV to read the video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / cap.get(cv2.CAP_PROP_FPS)
    interval = total_frames / no_of_frames_to_extract

    current_frame_number = 0
    extracted_count = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        # If the current frame number matches the interval, save the frame
        if current_frame_number % int(interval) == 0:
            frame_name = os.path.join(output_path, f"frame_{extracted_count:03}.jpg")
            cv2.imwrite(frame_name, frame)
            extracted_count += 1

            if extracted_count >= no_of_frames_to_extract:
                break

        current_frame_number += 1

    cap.release()


def main():
    args = parse_args()

    # Create output directory if not exists already
    os.makedirs(args.frames_dir_path, exist_ok=True)

    # Number of frames to be uniformly extracted
    no_of_frames_to_extract = 20

    videos_dir_path = args.video_input_dir_path

    for video in os.listdir(videos_dir_path):
        if video.endswith((".mp4", ".mov")):  # Check for supported video formats
            video_path = os.path.join(videos_dir_path, video)
            output_path = os.path.join(args.frames_dir_path, os.path.splitext(video)[0])
            os.makedirs(output_path, exist_ok=True)

            extract_uniform_frames(video_path, output_path, no_of_frames_to_extract)


if __name__ == "__main__":
    main()
