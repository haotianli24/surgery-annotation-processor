import xml.etree.ElementTree as ET
import cv2
import sys
import os
import numpy as np

def main():

    # processing annotation based on bash script 
    # usage: time python3 /home/howtian/medical_video_processing/blur.py "$input_video" "$output_video"
    if len(sys.argv) != 3:
        print("Usage: python blur.py <input_video> <output_video>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video '{input_video}' not found")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process XML file 
    xml_file = 'annotations.xml'
    if not os.path.exists(xml_file):
        print(f"Error: Annotations file '{xml_file}' not found in current directory")
        sys.exit(1)
    
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
    
    # Opening video for annotations (FFMPEG)
    cap = cv2.VideoCapture(input_video, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{input_video}'")
        sys.exit(1)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    # Create output video writer
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Cannot create output video file '{output_video}'")
        cap.release()
        sys.exit(1)

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
            blurred = cv2.GaussianBlur(frame, (71, 71), 0)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for region in regions:
                region_type = region[0]
                occluded = region[-1]
                
                # occluded = no blur needed 
                if occluded == 1:
                    continue
                
                if region_type == 'box':
                    _, x1, y1, x2, y2, _ = region
                    # ensure nothing goes out of bounds 
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        processed_regions += 1
                
                elif region_type == 'ellipse':
                    _, cx, cy, rx, ry, _ = region
                    
                    # Draw ellipse on mask
                    ellipse_center = (int(cx), int(cy))
                    ellipse_axes = (int(rx), int(ry))
                    cv2.ellipse(mask, ellipse_center, ellipse_axes, 0, 0, 360, 255, -1)
                    processed_regions += 1
            
            # Apply blur using mask
            frame = np.where(mask[:, :, None] == 255, blurred, frame)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    
    print(f"Processing complete!")
    print(f"Processed {frame_idx} frames")
    print(f"Applied blur to {processed_regions} regions")
    print(f"Output saved to: {output_video}")

if __name__ == "__main__":
    main()
