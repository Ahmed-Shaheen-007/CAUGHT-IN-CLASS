import cv2
import numpy as np
import face_recognition
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
import logging
from datetime import datetime
import hashlib
import os

logger = logging.getLogger(__name__)

class CameraWorker(QThread):
    """Worker thread for camera capture to prevent GUI freezing"""
    
    frame_ready = pyqtSignal(np.ndarray)  # Emit captured frame
    camera_error = pyqtSignal(str)  # Emit camera errors
    camera_status = pyqtSignal(str)  # Emit camera status
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.capture = None
        self.running = False
        self.frame_skip = 1  # Process every N frames
        
    def set_camera_index(self, index):
        """Set camera index"""
        self.camera_index = index
        
    def set_frame_skip(self, skip):
        """Set frame processing skip rate"""
        self.frame_skip = max(1, skip)
        
    def run(self):
        """Main camera loop"""
        try:
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            if not self.capture.isOpened():
                # Try default camera if specified camera fails
                self.capture = cv2.VideoCapture(0)
                
            if not self.capture.isOpened():
                raise Exception("Failed to open camera")
                
            # Set camera properties for better performance
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_status.emit("Camera started successfully")
            self.running = True
            
            frame_count = 0
            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    self.camera_error.emit("Failed to read frame")
                    continue
                    
                # Only emit every N frames for performance
                if frame_count % self.frame_skip == 0:
                    self.frame_ready.emit(frame.copy())
                    
                frame_count += 1
                self.msleep(33)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"Camera worker error: {e}")
            self.camera_error.emit(f"Camera error: {str(e)}")
        finally:
            if self.capture:
                self.capture.release()
            self.camera_status.emit("Camera stopped")
            
    def stop(self):
        """Stop camera capture"""
        self.running = False
        self.wait()

class DetectionWorker(QThread):
    """Worker thread for face detection and recognition"""
    
    detection_result = pyqtSignal(dict)  # Emit detection results
    detection_error = pyqtSignal(str)  # Emit detection errors
    status_update = pyqtSignal(str)  # Emit status updates
    
    def __init__(self, known_face_encodings, known_face_names):
        super().__init__()
        self.known_face_encodings = known_face_encodings
        self.known_face_names = known_face_names
        self.running = False
        self.session_seen_known = set()
        self.session_seen_unknown_hashes = set()
        
        # Frame processing control
        self.last_processing_time = 0
        self.processing_interval = 2  # Process every 1.5 seconds
        self.pending_frame = None
        self.frame_lock = QObject()  # Simple lock for frame access
        
    def update_known_faces(self, encodings, names):
        """Update known face encodings and names in real-time"""
        try:
            self.known_face_encodings = encodings.copy()
            self.known_face_names = names.copy()
            logger.info(f"Updated detection worker with {len(names)} known faces")
        except Exception as e:
            logger.error(f"Error updating known faces in detection worker: {e}")
            
    def set_processing_interval(self, interval):
        """Set the processing interval in seconds"""
        self.processing_interval = max(0.5, interval)  # Minimum 0.5 seconds
        logger.info(f"Detection processing interval set to {self.processing_interval} seconds")
        
    def queue_frame(self, frame):
        """Queue a frame for processing (replaces previous pending frame)"""
        try:
            # Clear previous pending frame to prevent memory leaks
            if self.pending_frame is not None:
                del self.pending_frame
                
            # Store new frame (make a copy to avoid reference issues)
            self.pending_frame = frame.copy()
            logger.debug("Frame queued for processing")
            
        except Exception as e:
            logger.error(f"Error queuing frame: {e}")
            
    def should_process_frame(self):
        """Check if enough time has passed to process the next frame"""
        current_time = datetime.now().timestamp()
        return (current_time - self.last_processing_time) >= self.processing_interval
        
    def reset_session(self):
        """Reset session tracking"""
        self.session_seen_known.clear()
        self.session_seen_unknown_hashes.clear()
        logger.info("Detection session reset")
        
    def run(self):
        """Main detection loop with controlled processing intervals"""
        self.running = True
        self.status_update.emit("Detection worker started")
        
        while self.running:
            try:
                # Check if we should process a frame
                if self.pending_frame is not None and self.should_process_frame():
                    # Process the pending frame
                    self.process_frame(self.pending_frame)
                    
                    # Update processing time
                    self.last_processing_time = datetime.now().timestamp()
                    
                    # Clear the processed frame to free memory
                    del self.pending_frame
                    self.pending_frame = None
                    
                    logger.debug("Frame processed and cleared from memory")
                
                # Sleep for a short interval to prevent busy waiting
                self.msleep(100)  # 100ms sleep
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                self.detection_error.emit(f"Detection loop error: {str(e)}")
                self.msleep(1000)  # Sleep longer on error
        
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        if not self.running:
            return
            
        try:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Optimize face detection by reducing frame size
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            
            # Detect faces in smaller frame for better performance
            face_locations = face_recognition.face_locations(small_frame, model="hog")
            
            # Scale locations back to original frame size
            face_locations = [(top*4, right*4, bottom*4, left*4) for top, right, bottom, left in face_locations]
            
            # Get encodings from original frame at detected locations
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Process each detected face
            detection_results = []
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                result = self._process_single_face(face_encoding, face_location, frame)
                if result:
                    detection_results.append(result)
                    
            # Emit results
            if detection_results:
                self.detection_result.emit({
                    'timestamp': datetime.now(),
                    'detections': detection_results,
                    'total_faces': len(face_locations)
                })
                
            # Clear intermediate variables to free memory
            del rgb_frame, small_frame, face_locations, face_encodings
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.detection_error.emit(f"Detection error: {str(e)}")
            
    def _process_single_face(self, face_encoding, face_location, frame):
        """Process a single detected face"""
        try:
            top, right, bottom, left = face_location
            
            # Crop face region for saving/display
            face_crop = frame[top:bottom, left:right]
            
            if len(self.known_face_encodings) == 0:
                # No known faces, mark as unknown
                return self._process_unknown_face(face_crop, face_location, face_encoding)
                
            # Compare with known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Use threshold for recognition accuracy
            if face_distances[best_match_index] < 0.6:
                name = self.known_face_names[best_match_index]
                return self._process_known_face(name, face_crop, face_location, face_encoding)
            else:
                return self._process_unknown_face(face_crop, face_location, face_encoding)
                
        except Exception as e:
            logger.error(f"Single face processing error: {e}")
            return None
            
    def _process_known_face(self, name, face_crop, face_location, face_encoding):
        """Process a known face detection"""
        # Check if already seen this session
        if name in self.session_seen_known:
            return None  # Already recorded this session
            
        # Mark as seen
        self.session_seen_known.add(name)
        
        # Generate unique hash for this detection
        detection_hash = self._generate_detection_hash(face_encoding, name)
        
        return {
            'type': 'known',
            'name': name,
            'face_crop': face_crop,
            'face_location': face_location,
            'detection_hash': detection_hash,
            'status': 'Present'
        }
        
    def _process_unknown_face(self, face_crop, face_location, face_encoding):
        """Process an unknown face detection"""
        # Generate unique hash for this unknown face
        unknown_hash = self._generate_detection_hash(face_encoding, "Unknown")
        
        # Check if this unknown face was already seen this session
        if unknown_hash in self.session_seen_unknown_hashes:
            return None  # Already recorded this unknown face this session
            
        # Mark as seen
        self.session_seen_unknown_hashes.add(unknown_hash)
        
        return {
            'type': 'unknown',
            'name': 'Unknown',
            'face_crop': face_crop,
            'face_location': face_location,
            'detection_hash': unknown_hash,
            'status': 'Unknown'
        }
        
    def _generate_detection_hash(self, face_encoding, name):
        """Generate a unique hash for face detection tracking"""
        # Use face encoding data and name to create unique hash
        hash_input = f"{name}_{face_encoding.tobytes().hex()[:32]}"
        return hashlib.md5(hash_input.encode()).hexdigest()
        
    def stop(self):
        """Stop detection worker"""
        self.running = False
        self.wait()
        
    def get_session_summary(self):
        """Get current session summary"""
        return {
            'known_faces': len(self.session_seen_known),
            'unknown_faces': len(self.session_seen_unknown_hashes),
            'known_names': list(self.session_seen_known),
            'total_detections': len(self.session_seen_known) + len(self.session_seen_unknown_hashes)
        }
        
    def get_processing_status(self):
        """Get current processing status"""
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_processing_time
        return {
            'processing_interval': self.processing_interval,
            'time_since_last_processing': time_since_last,
            'has_pending_frame': self.pending_frame is not None,
            'frames_processed': self.last_processing_time > 0
        } 