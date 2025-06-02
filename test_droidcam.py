import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test DroidCam Connection')
    parser.add_argument('--ip', type=str, default='192.168.1.103',
                        help='DroidCam IP address')
    parser.add_argument('--port', type=str, default='4747',
                        help='DroidCam port')
    
    args = parser.parse_args()
    
    # Construct URL
    url = f"http://{args.ip}:{args.port}/video"
    print(f"Connecting to DroidCam at: {url}")
    
    # Open video source
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print("Error: Could not connect to DroidCam")
        print("\nTroubleshooting tips:")
        print("1. Make sure DroidCam app is running on your phone")
        print("2. Verify the IP address and port shown in the DroidCam app")
        print("3. Ensure your phone and computer are on the same WiFi network")
        print("4. Check if any firewall is blocking the connection")
        print("5. Try accessing the DroidCam web interface in your browser:")
        print(f"   {url.replace('/video', '')}")
        return
    
    print("Connection successful!")
    print("Press 'q' to quit")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Read and display frames
    start_time = time.time()
    frame_count = 0
    fps = 0
    
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