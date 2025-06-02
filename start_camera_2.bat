@echo off
echo Starting YOLOv8 People Tracking System with Camera Index 2...
echo.
echo This script uses camera index 2, which might be your DroidCam USB connection
echo Web Interface: http://localhost:8080
echo.
echo Press Ctrl+C to stop the application
echo.

python people_counter.py --source 2 --port 8080

pause 