import cv2
import time
import sys

def print_camera_properties(cap):
    """Print detailed properties of the camera"""
    props = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_FOURCC, "Codec"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
        (cv2.CAP_PROP_GAIN, "Gain"),
        (cv2.CAP_PROP_CONVERT_RGB, "Convert RGB"),
        (cv2.CAP_PROP_BUFFERSIZE, "Buffer Size")
    ]
    
    print("Camera Properties:")
    for prop_id, prop_name in props:
        try:
            value = cap.get(prop_id)
            print(f"  {prop_name}: {value}")
        except:
            pass

def test_camera(index, show_image=True, delay=3):
    """Test camera with detailed information"""
    print(f"\n--- Testing camera index: {index} ---")
    
    # Try to open the camera
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"❌ Camera index {index} not available")
        return False
    
    print(f"✅ Camera index {index} is opened successfully!")
    
    # Print camera properties
    print_camera_properties(cap)
    
    # Read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to read a frame")
        cap.release()
        return False
    
    print(f"✅ Successfully read a frame of size: {frame.shape}")
    
    if show_image:
        # Display the frame
        window_name = f"Camera {index} Test"
        cv2.imshow(window_name, frame)
        print(f"Displaying image for {delay} seconds...")
        
        # Wait for specified seconds or until a key is pressed
        start_time = time.time()
        while (time.time() - start_time) < delay:
            if cv2.waitKey(100) != -1:
                break
        
        cv2.destroyWindow(window_name)
    
    cap.release()
    return True

def test_droidcam_usb():
    """Test if DroidCam is available through USB"""
    print("\n=== Testing DroidCam USB Connection ===")
    
    # First check if DroidCam is available as a regular camera
    for i in range(10):  # Test indices 0 through 9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Check if this might be DroidCam (simple heuristic)
                height, width = frame.shape[:2]
                if width >= 640 and "DroidCam" in cv2.__file__:
                    print(f"✅ Possible DroidCam found at index {i}")
                    print(f"   Frame size: {width}x{height}")
                else:
                    print(f"Camera found at index {i}, but might not be DroidCam")
                    print(f"   Frame size: {width}x{height}")
            cap.release()

def main():
    print("OpenCV Version:", cv2.__version__)
    print("Python Version:", sys.version)
    
    # Test all camera indices from 0 to 9
    print("\n=== Testing All Camera Indices ===")
    working_cameras = []
    
    for i in range(10):
        if test_camera(i, show_image=False):
            working_cameras.append(i)
    
    # Test DroidCam specifically
    test_droidcam_usb()
    
    # Summary
    if working_cameras:
        print(f"\n=== Summary ===")
        print(f"Working camera indices: {working_cameras}")
        print("To use a specific camera with the people counter:")
        print(f"python people_counter.py --source {working_cameras[0]}")
    else:
        print("\n❌ No working cameras found.")
    
    print("\nIf you're using DroidCam, make sure:")
    print("1. DroidCam app is running on your phone")
    print("2. DroidCam client is running on your computer")
    print("3. Your phone is connected via USB and USB debugging is enabled")
    print("4. Try using the IP address directly: --source http://YOUR_IP:PORT/video")
    print("   Example: --source http://192.168.1.103:4747/video")

if __name__ == "__main__":
    main() 