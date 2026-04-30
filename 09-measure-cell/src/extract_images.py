#!/usr/bin/env python3
"""
Extract frames from a video file at regular intervals and save as TIFF images.
"""

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path = "data",
    interval_seconds: int = 10,
) -> None:
    """
    Extract frames from a video at regular time intervals.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        interval_seconds: Extract a frame every n seconds (default: 10)
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames / fps:.2f} seconds")
    print(f"Extracting every {interval_seconds} seconds")

    # Calculate frame interval
    frame_interval: int = int(fps * interval_seconds)

    frame_number: int = 0
    frame_count: int = 0

    while True:
        ret: bool
        frame: cv2.Mat
        ret, frame = cap.read()

        if not ret:
            break

        # Extract frame at specified intervals
        if frame_number % frame_interval == 0:
            output_path: str = str(Path(output_dir) / f"img_{frame_count:04d}.tif")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
            frame_count += 1

        frame_number += 1

    cap.release()
    print(f"\nExtraction complete! Total frames saved: {frame_count}")


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Extract frames from a video file at regular intervals"
    )
    parser.add_argument(
        "video",
        help="Path to the video file"
    )
    parser.add_argument(
        "-o", "--output",
        default="data",
        help="Output directory for frames (default: data)"
    )
    parser.add_argument(
        "-i", "--interval",
        type=int,
        default=10,
        help="Extract a frame every n seconds (default: 10)"
    )

    args: argparse.Namespace = parser.parse_args()

    extract_frames(args.video, args.output, args.interval)
