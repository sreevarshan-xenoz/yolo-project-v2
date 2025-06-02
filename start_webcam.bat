@echo off
echo Starting YOLOv8 People Tracking System with Webcam...
echo.
echo Web Interface: http://localhost:8080
echo.
echo Press Ctrl+C to stop the application
echo.

python people_counter.py --source 0 --port 8080

pause 