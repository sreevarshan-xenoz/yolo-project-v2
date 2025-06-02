import cv2

def test_camera(index):
    print(f"Testing camera index: {index}")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Camera index {index} not available")
        return False
    
    print(f"Camera index {index} is working!")
    ret, frame = cap.read()
    if ret:
        print(f"Successfully read a frame of size: {frame.shape}")
        
    cap.release()
    return True

# Test camera indices 0 to 5
print("Testing all camera indices from 0 to 5...")
working_cameras = []

for i in range(6):
    if test_camera(i):
        working_cameras.append(i)

if working_cameras:
    print(f"\nWorking camera indices: {working_cameras}")
    print("Use one of these indices with the --source parameter")
else:
    print("\nNo working cameras found.")
    print("If you're using DroidCam, make sure:")
    print("1. DroidCam app is running on your phone")
    print("2. Your phone and computer are on the same WiFi network")
    print("3. Try using the IP address and port directly: --source http://YOUR_IP:PORT/video")
    print("   Example: --source http://192.168.1.103:4747/video") 