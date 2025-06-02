import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import threading
import queue
import time
import argparse
import os
from flask import Flask, Response, render_template, request, jsonify
import socket
import logging
import torch
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('people_counter')

# ================= CONFIGURATION =================
# Note: YOLOv11 might not be available yet; use YOLOv8 if needed (e.g., "yolov8l.pt")
MODEL_PATH = "yolov8l.pt"  # YOLOv8 Large as fallback
FRAME_WIDTH, FRAME_HEIGHT = 640, 480  # Balanced resolution for streaming
SKIP_FRAMES = 1  # Process every 2nd frame initially (adaptive)
LINE_Y_PERCENT = 0.55  # Line position as percentage of frame height
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
# =================================================

# Flask app for web interface
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Global variables for sharing data between threads
frame_buffer = None
frame_lock = threading.Lock()
analytics_data = {
    'counts': defaultdict(int),
    'fps': 0,
    'processing_time': 0,
    'device': 'unknown',
    'model_name': os.path.basename(MODEL_PATH),
    'status': 'initializing',
    'current_count': 0
}
analytics_lock = threading.Lock()

class OptimizedVideoCapture:
    """Enhanced video capture with multiple optimization options and automatic reconnection"""
    def __init__(self, source=0, width=640, height=480, buffer_size=2):
        self.source = source
        self.width = width
        self.height = height
        self.buffer_size = buffer_size
        self.q = queue.Queue(maxsize=buffer_size)
        self.running = False
        self.thread = None
        self.cap = None
        self.connect_attempts = 0
        self.max_connect_attempts = 5
        self.reconnect_delay = 2  # seconds
        
    def start(self):
        """Start capture thread with automatic reconnection"""
        self.running = True
        self.thread = threading.Thread(target=self._reader)
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def _connect(self):
        """Try to connect to the video source"""
        try:
            # Close existing capture if any
            if self.cap is not None:
                self.cap.release()
                
            # Check if source is DroidCam URL
            if isinstance(self.source, str) and (self.source.startswith('http') or ':' in self.source):
                if 'http' not in self.source:
                    self.source = f'http://{self.source}/video'
                logger.info(f"Connecting to DroidCam at {self.source}")
            
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception("Failed to open video source")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # For IP cameras/DroidCam: set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.connect_attempts = 0
            logger.info(f"Successfully connected to video source: {self.source}")
            return True
        except Exception as e:
            self.connect_attempts += 1
            logger.error(f"Connection attempt {self.connect_attempts} failed: {str(e)}")
            return False

    def _reader(self):
        """Background thread for reading frames"""
        last_reconnect_time = 0
        
        while self.running:
            # Try to connect if not connected
            if self.cap is None or not self.cap.isOpened():
                if time.time() - last_reconnect_time > self.reconnect_delay:
                    if self._connect():
                        last_reconnect_time = time.time()
                    else:
                        if self.connect_attempts >= self.max_connect_attempts:
                            logger.error("Maximum connection attempts reached")
                            self.running = False
                            break
                        time.sleep(self.reconnect_delay)
                        last_reconnect_time = time.time()
                continue
                
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("Failed to read frame, will attempt reconnection")
                if self.cap is not None:
                    self.cap.release()
                self.cap = None
                continue
                
            # Resize frame
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Update queue
            if not self.q.full():
                self.q.put(frame)
            else:
                # Discard oldest frame to maintain freshness
                try:
                    self.q.get_nowait()
                    self.q.put(frame)
                except queue.Empty:
                    pass

    def read(self):
        """Read a frame from the queue"""
        try:
            return True, self.q.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self):
        """Release resources"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()

def get_optimal_device():
    """Determine the optimal device for inference"""
    if torch.cuda.is_available():
        return "cuda", True  # Device, half precision
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", False  # Apple Silicon, no half precision
    else:
        return "cpu", False

def optimize_cv2():
    """Apply OpenCV optimizations"""
    cv2.setUseOptimized(True)
    
    # Set number of threads based on CPU cores
    num_cores = psutil.cpu_count(logical=False)
    if num_cores:
        optimal_threads = max(1, min(num_cores - 1, 4))  # Leave at least one core free
        cv2.setNumThreads(optimal_threads)
        logger.info(f"OpenCV using {optimal_threads} threads")

def process_frames(video_source):
    """Main processing function to run in a separate thread"""
    global frame_buffer, analytics_data
    
    # Initialize video capture
    cap = OptimizedVideoCapture(video_source, FRAME_WIDTH, FRAME_HEIGHT, buffer_size=2).start()
    
    # Optimize OpenCV
    optimize_cv2()
    
    # Load YOLOv8 model with hardware acceleration
    try:
        device, half_precision = get_optimal_device()
        model = YOLO(MODEL_PATH)
        model.to(device)
        
        with analytics_lock:
            analytics_data['device'] = device
            analytics_data['status'] = 'running'
            
        logger.info(f"Model loaded on {device} with half precision: {half_precision}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        with analytics_lock:
            analytics_data['status'] = f"error: {str(e)}"
        return

    # Tracking initialization
    crossed_ids = set()
    frame_counter = 0
    fps_update_interval = 10  # Update FPS every 10 frames
    processing_times = []
    
    # Calculate line position
    line_y = int(FRAME_HEIGHT * LINE_Y_PERCENT)
    
    start_time = time.time()
    
    try:
        while True:
            loop_start = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                time.sleep(0.1)
                continue
            
            frame_counter += 1
            
            # Skip frames if needed
            if frame_counter % (SKIP_FRAMES + 1) != 0:
                # Still update the display frame
                with frame_lock:
                    frame_buffer = frame.copy()
                continue
            
            # Process frame
            try:
                infer_start = time.time()
                
                # Run inference with optimized settings
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],  # Only track people
                    conf=CONFIDENCE_THRESHOLD,
                    iou=IOU_THRESHOLD,
                    device=device,
                    verbose=False
                )
                
                infer_time = time.time() - infer_start
                processing_times.append(infer_time)
                
                # Process results
                annotated_frame = frame.copy()
                
                # Draw counting line
                cv2.line(annotated_frame, (50, line_y), (FRAME_WIDTH-50, line_y), (0, 0, 255), 2)
                
                current_count = 0
                
                if results and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
                    track_ids = boxes.id.int().cpu().tolist()
                    
                    current_count = len(track_ids)
                    
                    for (x1, y1, x2, y2), track_id in zip(xyxy, track_ids):
                        # Calculate centroid
                        cy = (y1 + y2) // 2
                        
                        # Check if crossing the line
                        if cy > line_y and track_id not in crossed_ids:
                            crossed_ids.add(track_id)
                            with analytics_lock:
                                analytics_data['counts']["person"] += 1
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw ID
                        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate FPS
                if frame_counter % fps_update_interval == 0:
                    elapsed = time.time() - start_time
                    fps = fps_update_interval / elapsed if elapsed > 0 else 0
                    avg_processing = sum(processing_times) / len(processing_times) if processing_times else 0
                    
                    with analytics_lock:
                        analytics_data['fps'] = round(fps, 1)
                        analytics_data['processing_time'] = round(avg_processing * 1000, 1)  # ms
                        analytics_data['current_count'] = current_count
                    
                    processing_times = []
                    start_time = time.time()
                
                # Display FPS and counts on frame
                with analytics_lock:
                    fps = analytics_data['fps']
                    person_count = analytics_data['counts']["person"]
                
                cv2.putText(annotated_frame, f"FPS: {fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                           
                cv2.putText(annotated_frame, f"Count: {person_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                           
                cv2.putText(annotated_frame, f"Current: {current_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Update the frame buffer
                with frame_lock:
                    frame_buffer = annotated_frame.copy()
                
                # Adaptive frame skipping based on processing time
                if infer_time > 0.1:  # If processing takes >100ms
                    new_skip = min(5, SKIP_FRAMES + 1)  # Increase skip, max 5
                    if new_skip != SKIP_FRAMES:
                        logger.info(f"Adjusting frame skip to {new_skip} due to processing time {infer_time:.3f}s")
                        SKIP_FRAMES = new_skip
                elif infer_time < 0.05 and SKIP_FRAMES > 1:  # If processing is fast and we're skipping
                    SKIP_FRAMES = max(1, SKIP_FRAMES - 1)  # Decrease skip, min 1
                    logger.info(f"Adjusting frame skip to {SKIP_FRAMES} due to processing time {infer_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                with analytics_lock:
                    analytics_data['status'] = f"warning: {str(e)}"
            
            # Sleep to maintain reasonable CPU usage
            loop_time = time.time() - loop_start
            if loop_time < 0.01:  # If processing was very fast
                time.sleep(0.01)  # Prevent CPU overuse
                
    except Exception as e:
        logger.error(f"Processing thread error: {e}")
        with analytics_lock:
            analytics_data['status'] = f"error: {str(e)}"
    finally:
        cap.release()

def generate_frames():
    """Generator function for streaming frames"""
    while True:
        with frame_lock:
            if frame_buffer is not None:
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                # Yield the frame in multipart response format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Maintain reasonable frame rate for streaming
        time.sleep(0.03)  # ~30 FPS max

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analytics')
def get_analytics():
    """API endpoint for analytics data"""
    with analytics_lock:
        return jsonify(analytics_data)

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update settings via API"""
    global LINE_Y_PERCENT, CONFIDENCE_THRESHOLD, SKIP_FRAMES
    
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    if 'line_position' in data:
        LINE_Y_PERCENT = max(0.1, min(0.9, float(data['line_position'])))
    
    if 'confidence' in data:
        CONFIDENCE_THRESHOLD = max(0.1, min(0.9, float(data['confidence'])))
    
    if 'skip_frames' in data:
        SKIP_FRAMES = max(0, min(10, int(data['skip_frames'])))
    
    return jsonify({
        "status": "success",
        "settings": {
            "line_position": LINE_Y_PERCENT,
            "confidence": CONFIDENCE_THRESHOLD,
            "skip_frames": SKIP_FRAMES
        }
    })

@app.route('/reset', methods=['POST'])
def reset_counter():
    """Reset people counter"""
    with analytics_lock:
        analytics_data['counts'] = defaultdict(int)
    return jsonify({"status": "success"})

def get_ip_address():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def create_html_template():
    """Create HTML template if it doesn't exist"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 People Counter</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
            padding: 20px;
        }
        .video-container {
            position: relative;
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
        }
        .video-feed {
            width: 100%;
            height: auto;
            display: block;
        }
        .card {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.5);
            margin-bottom: 20px;
        }
        .card-header {
            background-color: #2d2d2d;
            color: #ffffff;
            font-weight: bold;
        }
        .btn-primary {
            background-color: #007bff;
            border-color: #007bff;
        }
        .btn-danger {
            background-color: #dc3545;
            border-color: #dc3545;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #00bfff;
        }
        .slider-container {
            padding: 10px 0;
        }
        .settings-label {
            margin-bottom: 5px;
            display: block;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 5px;
        }
        .status-running {
            background-color: #28a745;
        }
        .status-warning {
            background-color: #ffc107;
        }
        .status-error {
            background-color: #dc3545;
        }
        .status-initializing {
            background-color: #6c757d;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="text-center">YOLOv8 People Counter</h1>
                <div class="d-flex justify-content-center align-items-center">
                    <span id="status-indicator" class="status-indicator status-initializing"></span>
                    <span id="status-text">Initializing...</span>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <div class="video-container">
                    <img src="{{ url_for('video_feed') }}" class="video-feed" alt="Video Feed">
                </div>
                
                <div class="card">
                    <div class="card-header">Settings</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-4">
                                <div class="slider-container">
                                    <label for="linePosition" class="settings-label">Counting Line Position</label>
                                    <input type="range" class="form-range" min="0.1" max="0.9" step="0.05" id="linePosition" value="0.55">
                                    <small class="text-muted" id="linePositionValue">55%</small>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="slider-container">
                                    <label for="confidence" class="settings-label">Confidence Threshold</label>
                                    <input type="range" class="form-range" min="0.1" max="0.9" step="0.05" id="confidence" value="0.5">
                                    <small class="text-muted" id="confidenceValue">0.50</small>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="slider-container">
                                    <label for="skipFrames" class="settings-label">Frame Skip</label>
                                    <input type="range" class="form-range" min="0" max="5" step="1" id="skipFrames" value="1">
                                    <small class="text-muted" id="skipFramesValue">1</small>
                                </div>
                            </div>
                        </div>
                        <div class="row mt-3">
                            <div class="col-12 d-flex justify-content-between">
                                <button id="saveSettings" class="btn btn-primary">Save Settings</button>
                                <button id="resetCounter" class="btn btn-danger">Reset Counter</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">Analytics</div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6 mb-3">
                                <div class="text-center">
                                    <div>People Count</div>
                                    <div id="peopleCount" class="stat-value">0</div>
                                </div>
                            </div>
                            <div class="col-6 mb-3">
                                <div class="text-center">
                                    <div>Current People</div>
                                    <div id="currentCount" class="stat-value">0</div>
                                </div>
                            </div>
                            <div class="col-6 mb-3">
                                <div class="text-center">
                                    <div>FPS</div>
                                    <div id="fps" class="stat-value">0.0</div>
                                </div>
                            </div>
                            <div class="col-6 mb-3">
                                <div class="text-center">
                                    <div>Processing Time</div>
                                    <div id="processingTime" class="stat-value">0 ms</div>
                                </div>
                            </div>
                        </div>
                        <hr>
                        <div class="row">
                            <div class="col-6">
                                <div>Model:</div>
                                <div id="modelName" class="text-info">yolov8l.pt</div>
                            </div>
                            <div class="col-6">
                                <div>Device:</div>
                                <div id="deviceName" class="text-info">cpu</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update settings display
        document.getElementById('linePosition').addEventListener('input', function() {
            document.getElementById('linePositionValue').textContent = Math.round(this.value * 100) + '%';
        });
        
        document.getElementById('confidence').addEventListener('input', function() {
            document.getElementById('confidenceValue').textContent = parseFloat(this.value).toFixed(2);
        });
        
        document.getElementById('skipFrames').addEventListener('input', function() {
            document.getElementById('skipFramesValue').textContent = this.value;
        });
        
        // Save settings
        document.getElementById('saveSettings').addEventListener('click', function() {
            const settings = {
                line_position: parseFloat(document.getElementById('linePosition').value),
                confidence: parseFloat(document.getElementById('confidence').value),
                skip_frames: parseInt(document.getElementById('skipFrames').value)
            };
            
            fetch('/settings', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(settings)
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    alert('Settings updated successfully!');
                } else {
                    alert('Error updating settings: ' + data.message);
                }
            })
            .catch(error => {
                alert('Error: ' + error);
            });
        });
        
        // Reset counter
        document.getElementById('resetCounter').addEventListener('click', function() {
            fetch('/reset', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    document.getElementById('peopleCount').textContent = '0';
                }
            });
        });
        
        // Update analytics
        function updateAnalytics() {
            fetch('/analytics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('peopleCount').textContent = data.counts.person || 0;
                    document.getElementById('currentCount').textContent = data.current_count || 0;
                    document.getElementById('fps').textContent = data.fps || 0;
                    document.getElementById('processingTime').textContent = data.processing_time + ' ms';
                    document.getElementById('modelName').textContent = data.model_name;
                    document.getElementById('deviceName').textContent = data.device;
                    
                    // Update status indicator
                    const statusIndicator = document.getElementById('status-indicator');
                    const statusText = document.getElementById('status-text');
                    
                    statusIndicator.className = 'status-indicator';
                    
                    if (data.status === 'running') {
                        statusIndicator.classList.add('status-running');
                        statusText.textContent = 'Running';
                    } else if (data.status.startsWith('warning')) {
                        statusIndicator.classList.add('status-warning');
                        statusText.textContent = data.status;
                    } else if (data.status.startsWith('error')) {
                        statusIndicator.classList.add('status-error');
                        statusText.textContent = data.status;
                    } else {
                        statusIndicator.classList.add('status-initializing');
                        statusText.textContent = data.status;
                    }
                })
                .catch(error => {
                    console.error('Error fetching analytics:', error);
                });
        }
        
        // Update analytics every second
        setInterval(updateAnalytics, 1000);
        
        // Initial update
        updateAnalytics();
    </script>
</body>
</html>
    """
    
    with open('templates/index.html', 'w') as f:
        f.write(html_content)
    
    logger.info("HTML template created")

def main():
    global MODEL_PATH
    
    parser = argparse.ArgumentParser(description='YOLOv8 People Counter with DroidCam')
    parser.add_argument('--source', type=str, default='http://192.168.1.103:4747/video',
                        help='Video source (DroidCam URL, webcam index, or video file)')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port for the web interface')
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to YOLO model')
    
    args = parser.parse_args()
    
    # Update model path if specified
    if args.model != MODEL_PATH:
        MODEL_PATH = args.model
        with analytics_lock:
            analytics_data['model_name'] = os.path.basename(MODEL_PATH)
    
    # Create HTML template
    create_html_template()
    
    # Start processing thread
    processing_thread = threading.Thread(target=process_frames, args=(args.source,))
    processing_thread.daemon = True
    processing_thread.start()
    
    # Get local IP
    ip = get_ip_address()
    
    # Start Flask app
    logger.info(f"Starting web interface at http://{ip}:{args.port}/")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

if __name__ == '__main__':
    main() 