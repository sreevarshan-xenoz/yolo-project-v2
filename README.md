# YOLOv8 People Tracking System with DroidCam

A real-time people tracking and counting system that uses your webcam or Android phone as a camera source via DroidCam and provides a web interface for monitoring.

## Features

- **Camera Support**: Use your laptop's webcam or Android phone via DroidCam (USB or WiFi)
- **Real-time Object Detection**: Powered by YOLOv8 with hardware acceleration
- **People Counting**: Count people crossing a virtual line
- **Web Interface**: Monitor the video feed and statistics in real-time
- **Adaptive Performance**: Automatically adjusts processing parameters for optimal performance

## Prerequisites

- Python 3.8 or higher
- If using DroidCam:
  - DroidCam app installed on your Android phone
  - DroidCam client installed on your computer (for USB connection)
  - Computer and Android phone connected to the same WiFi network (for WiFi connection)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sreevarshan-xenoz/yolo-project-v2.git
   cd yolo-project-v2
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model (if not already included):
   ```
   # The script will download the model automatically if not found
   ```

## Camera Setup Options

### Option 1: Using Your Laptop's Built-in Webcam

1. Simply run:
   ```
   start_droidcam_usb.bat
   ```
   (This will use camera index 0, which is typically the built-in webcam)

### Option 2: Using DroidCam via USB

1. Install DroidCam app on your Android device
2. Install DroidCam client on your computer
3. Connect your phone via USB and start the DroidCam app
4. Start the DroidCam client on your computer
5. Run one of these scripts:
   ```
   start_droidcam_usb.bat   # Uses camera index 0
   ```
   or
   ```
   start_camera_2.bat       # Uses camera index 2
   ```

### Option 3: Using DroidCam via WiFi

1. Install DroidCam app on your Android device
2. Connect your phone to the same WiFi network as your computer
3. Open the DroidCam app and note the IP address and port (e.g., 192.168.1.103:4747)
4. Run:
   ```
   python people_counter.py --source http://YOUR_IP:PORT/video --port 8080
   ```
   Replace YOUR_IP:PORT with the values shown in the DroidCam app

## Testing Your Camera

If you're having trouble with camera connections, run the camera test script:

```
python camera_test_detailed.py
```

This will:
- Check all available camera indices (0-9)
- Show which cameras are working
- Display detailed properties for each camera
- Recommend which camera index to use

## Using the Web Interface

1. After starting the application, open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

2. The web interface provides:
   - Live video feed with detection visualization
   - People count statistics
   - Performance metrics (FPS, processing time)
   - Adjustable settings:
     - Counting line position
     - Confidence threshold
     - Frame skip rate

## Troubleshooting

### Camera Connection Issues

- **Built-in Webcam**: If your laptop's webcam isn't working, check if other applications can access it
- **DroidCam USB**: 
  - Make sure USB debugging is enabled on your phone
  - Check if the DroidCam client shows a connected status
  - Try different camera indices (0, 1, 2) using the test script
- **DroidCam WiFi**: 
  - Ensure your phone and computer are on the same WiFi network
  - Verify the IP address and port in the DroidCam app
  - Try accessing the DroidCam web interface in your browser: http://YOUR_IP:PORT

### Performance Issues

- Try reducing the frame skip rate or using a smaller model
- Close other applications that might be using the GPU
- Check the FPS in the web interface; if it's below 5, consider lowering resolution

### Model Errors

- Make sure the model file exists and is accessible
- If you see CUDA errors, try running with CPU only: add `--device cpu` to the command

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [DroidCam](https://www.dev47apps.com/) for the camera app
- [Flask](https://flask.palletsprojects.com/) for the web framework
