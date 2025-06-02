import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test DroidCam Connection')
    parser.add_argument('--source', type=str, default='http://192.168.1.103:4747/video',
                        help='DroidCam URL or webcam index')
    
    args = parser.parse_args()
    
    print(f"Connecting to video source: {args.source}")
    
    # Open video source
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    print("Connection successful!")
    print("Press 'q' to quit")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Read and display frames
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        frame_count += 1
        elapsed_time = time.time() - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # Display FPS on frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow('DroidCam Test', frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 