@echo off
echo Starting YOLOv8 People Tracking System with DroidCam USB...
echo.
echo Make sure DroidCam is connected via USB and the client is running
echo Web Interface: http://localhost:8080
echo.
echo Press Ctrl+C to stop the application
echo.

REM Using camera index 0 which is working according to tests
python people_counter.py --source 0 --port 8080

pause 