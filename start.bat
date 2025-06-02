@echo off
echo Starting YOLOv8 People Tracking System...
echo.
echo DroidCam URL: http://192.168.1.103:4747/video
echo Web Interface: http://localhost:8080
echo.
echo Press Ctrl+C to stop the application
echo.

python people_counter.py --source http://192.168.1.103:4747/video --port 8080

pause 