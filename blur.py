import xml.etree.ElementTree as ET
import cv2
import sys
import os
import numpy as np

def process_video(input_video, output_video, xml_file='annotations.xml'):
    """Process a single video with blurring based on annotations."""
    print(f"\n{'='*80}")
    print(f"Processing video: {os.path.basename(input_video)}")
    print(f"{'='*80}")
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process XML file 
    if not os.path.exists(xml_file):
        print(f"Error: Annotations file '{xml_file}' not found")
        return False
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = {}
    
    # Parse rectangular bounding boxes
    for track in root.findall('.//track'):
        for box in track.findall('.//box'):
            frame = int(box.attrib['frame'])
            occluded = int(box.attrib['occluded'])
            x1 = int(float(box.attrib['xtl']))
            y1 = int(float(box.attrib['ytl']))
            x2 = int(float(box.attrib['xbr']))
            y2 = int(float(box.attrib['ybr']))
            region = ('box', x1, y1, x2, y2, occluded)
            annotations.setdefault(frame, []).append(region)
        
        # Parse ellipses
        for ellipse in track.findall('.//ellipse'):
            frame = int(ellipse.attrib['frame'])
            occluded = int(ellipse.attrib['occluded'])
            cx = float(ellipse.attrib['cx'])  # center x
            cy = float(ellipse.attrib['cy'])  # center y
            rx = float(ellipse.attrib['rx'])  # radius x
            ry = float(ellipse.attrib['ry'])  # radius y
            region = ('ellipse', cx, cy, rx, ry, occluded)
            annotations.setdefault(frame, []).append(region)
        
        # Parse polygons
        for polygon in track.findall('.//polygon'):
            frame = int(polygon.attrib['frame'])
            occluded = int(polygon.attrib.get('occluded', '0'))
            points_str = polygon.attrib['points']
            points = []
            for pair in points_str.strip().split(';'):
                if not pair:
                    continue
                x_str, y_str = pair.split(',')
                x = int(float(x_str))
                y = int(float(y_str))
                points.append((x, y))
            region = ('polygon', points, occluded)
            annotations.setdefault(frame, []).append(region)
    
    # Opening video for annotations (FFMPEG)
    cap = cv2.VideoCapture(input_video, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_video}'")
        return False
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Create output video writer
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file '{output_video}'")
        cap.release()
        return False

    # Process frames
    frame_idx = 0
    processed_regions = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply blurring to annotated regions
        regions = annotations.get(frame_idx, [])
        if regions:

            for region in regions:
                region_type = region[0]
                occluded = region[-1]
                if occluded == 1:
                    continue

                mask = np.zeros((height, width), dtype=np.uint8)

                # -----------------------
                # BOX
                # -----------------------
                if region_type == 'box':
                    _, x1, y1, x2, y2, _ = region

                    # Clip bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue

                    k = int(0.35 * max(w, h))
                    if k % 2 == 0:
                        k += 1

                    blurred_full = cv2.GaussianBlur(frame, (k, k), 0)

                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                    frame = np.where(mask[:, :, None] == 255, blurred_full, frame)
                    processed_regions += 1

                # -----------------------
                # ELLIPSE
                # -----------------------
                elif region_type == 'ellipse':
                    _, cx, cy, rx, ry, _ = region
                    cx_i, cy_i = int(cx), int(cy)
                    rx_i, ry_i = int(rx), int(ry)

                    w = rx_i * 2
                    h = ry_i * 2

                    k = int(0.35 * max(w, h))
                    if k % 2 == 0:
                        k += 1

                    blurred_full = cv2.GaussianBlur(frame, (k, k), 0)

                    cv2.ellipse(mask, (cx_i, cy_i), (rx_i, ry_i), 
                                0, 0, 360, 255, -1)

                    frame = np.where(mask[:, :, None] == 255, blurred_full, frame)
                    processed_regions += 1

                # -----------------------
                # POLYGON
                # -----------------------
                elif region_type == 'polygon':
                    _, pts, _ = region
                    pts_np = np.array(pts, dtype=np.int32)

                    # Compute size for dynamic blur
                    x_min = np.min(pts_np[:,0])
                    x_max = np.max(pts_np[:,0])
                    y_min = np.min(pts_np[:,1])
                    y_max = np.max(pts_np[:,1])
                    w = x_max - x_min
                    h = y_max - y_min

                    if w <= 0 or h <= 0:
                        continue

                    k = int(0.35 * max(w, h))
                    if k % 2 == 0:
                        k += 1

                    blurred_full = cv2.GaussianBlur(frame, (k, k), 0)

                    cv2.fillPoly(mask, [pts_np], 255)

                    frame = np.where(mask[:, :, None] == 255, blurred_full, frame)
                    processed_regions += 1

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    print(f"Processing complete!")
    print(f"Processed {frame_idx} frames")
    print(f"Applied blur to {processed_regions} regions")
    print(f"Output saved to: {output_video}")
    
    return True

def main():
    # processing annotation based on bash script 
    # usage: python3 blur.py <input_video1> <output_video1> <annotation_file1> [<input_video2> <output_video2> <annotation_file2> ...]
    
    if len(sys.argv) < 4 or (len(sys.argv) - 1) % 3 != 0:
        print("Usage: python blur.py <input_video1> <output_video1> <annotation_file1> [<input_video2> <output_video2> <annotation_file2> ...]")
        print("\nExample (single video):")
        print("  python blur.py input1.mp4 output1.mp4 annotations1.xml")
        print("\nExample (multiple videos):")
        print("  python blur.py input1.mp4 output1.mp4 annotations1.xml input2.mp4 output2.mp4 annotations2.xml")
        sys.exit(1)
    
    # Parse video triplets from command line arguments
    video_tasks = []
    for i in range(1, len(sys.argv), 3):
        input_video = sys.argv[i]
        output_video = sys.argv[i + 1]
        annotation_file = sys.argv[i + 2]
        video_tasks.append((input_video, output_video, annotation_file))
    
    print(f"\n{'#'*80}")
    print(f"# Starting batch video processing")
    print(f"# Total videos to process: {len(video_tasks)}")
    print(f"{'#'*80}")
    
    # Process each video task
    success_count = 0
    failure_count = 0
    
    for idx, (input_video, output_video, annotation_file) in enumerate(video_tasks, 1):
        print(f"\n[{idx}/{len(video_tasks)}] Starting video processing...")
        print(f"  Input: {input_video}")
        print(f"  Output: {output_video}")
        print(f"  Annotations: {annotation_file}")
        
        if process_video(input_video, output_video, annotation_file):
            success_count += 1
        else:
            failure_count += 1
            print(f"Failed to process: {input_video}")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"# Batch processing complete")
    print(f"# Successfully processed: {success_count}/{len(video_tasks)}")
    if failure_count > 0:
        print(f"# Failed: {failure_count}/{len(video_tasks)}")
    print(f"{'#'*80}\n")
    
    # Exit with error code if any failures
    if failure_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()