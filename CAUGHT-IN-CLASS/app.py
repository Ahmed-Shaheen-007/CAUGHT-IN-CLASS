#!/usr/bin/env python3
"""
Enhanced Face Attendance System v3.0
Features:
- Multi-threaded camera capture and face detection
- Thumbnail display for detected faces
- Audio notifications
- Dark/light theme switching
- Session-based attendance tracking
- Master Excel logging
- Robust image handling
"""

import sys
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QSplitter, QToolBar, QStatusBar, QProgressBar,
                            QMessageBox, QFileDialog, QMenuBar, QAction,
                            QFrame, QScrollArea, QDialog, QSpinBox, QCheckBox,
                            QComboBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon

# Import our custom modules
from workers import CameraWorker, DetectionWorker
from face_manager import FaceManager
from logger import AttendanceLogger
from config import ConfigManager
from audio_manager import AudioManager
from thumbnail_widget import ThumbnailPanel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SettingsDialog(QDialog):
    """Settings dialog for camera and application configuration"""
    
    def __init__(self, config_manager, parent=None):
        super().__init__(parent)
        self.config_manager = config_manager
        self.setup_ui()
        self.load_current_settings()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Camera settings group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QFormLayout(camera_group)
        
        self.camera_index_spin = QSpinBox()
        self.camera_index_spin.setRange(0, 10)
        self.camera_index_spin.setValue(0)
        camera_layout.addRow("Camera Index:", self.camera_index_spin)
        
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(1)
        self.frame_skip_spin.setToolTip("Process every N frames (higher = better performance)")
        camera_layout.addRow("Frame Skip:", self.frame_skip_spin)
        
        self.detection_interval_spin = QSpinBox()
        self.detection_interval_spin.setRange(5, 50)  # 0.5 to 5.0 seconds
        self.detection_interval_spin.setValue(15)  # 1.5 seconds
        self.detection_interval_spin.setSuffix(" (×0.1s)")
        self.detection_interval_spin.setToolTip("Interval between face detection processing (higher = better performance)")
        camera_layout.addRow("Detection Interval:", self.detection_interval_spin)
        
        layout.addWidget(camera_group)
        
        # Audio settings group
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QFormLayout(audio_group)
        
        self.sound_checkbox = QCheckBox("Enable Sound Notifications")
        self.sound_checkbox.setChecked(True)
        audio_layout.addRow(self.sound_checkbox)
        
        layout.addWidget(audio_group)
        
        # Theme settings group
        theme_group = QGroupBox("Interface Settings")
        theme_layout = QFormLayout(theme_group)
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["light", "dark"])
        theme_layout.addRow("Theme:", self.theme_combo)
        
        self.thumbnail_size_spin = QSpinBox()
        self.thumbnail_size_spin.setRange(60, 120)
        self.thumbnail_size_spin.setValue(80)
        theme_layout.addRow("Thumbnail Size:", self.thumbnail_size_spin)
        
        layout.addWidget(theme_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)
        
        button_layout.addStretch()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
    def load_current_settings(self):
        """Load current configuration settings"""
        config = self.config_manager.get_config_summary()
        
        self.camera_index_spin.setValue(config.get('camera_index', 0))
        self.frame_skip_spin.setValue(config.get('frame_skip', 1))
        self.detection_interval_spin.setValue(int(config.get('detection_interval', 1.5) * 10))  # Convert to 0.1s units
        self.sound_checkbox.setChecked(config.get('sound_enabled', True))
        self.theme_combo.setCurrentText(config.get('theme', 'light'))
        self.thumbnail_size_spin.setValue(config.get('thumbnail_size', 80))
        
    def save_settings(self):
        """Save configuration settings"""
        try:
            self.config_manager.set('camera_index', self.camera_index_spin.value())
            self.config_manager.set('frame_skip', self.frame_skip_spin.value())
            self.config_manager.set('detection_interval', self.detection_interval_spin.value() / 10.0)  # Convert from 0.1s units
            self.config_manager.set('sound_enabled', self.sound_checkbox.isChecked())
            self.config_manager.set('theme', self.theme_combo.currentText())
            self.config_manager.set('thumbnail_size', self.thumbnail_size_spin.value())
            
            QMessageBox.information(self, "Settings", "Settings saved successfully!")
            self.accept()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save settings: {str(e)}")
            
    def reset_to_defaults(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings", 
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.config_manager.reset_to_defaults()
            self.load_current_settings()

class AttendanceApp(QMainWindow):
    """Main application window for Face Attendance System"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.config_manager = ConfigManager()
        self.face_manager = FaceManager()
        self.attendance_logger = AttendanceLogger()
        self.audio_manager = AudioManager()
        
        # Initialize workers
        self.camera_worker = None
        self.detection_worker = None
        
        # Session tracking
        self.session_records = []
        self.is_running = False
        
        # Track latest unknown frame for registration
        self.latest_unknown_frame = None
        self.latest_unknown_detection = None
        
        # Status update timer
        self.status_update_timer = QTimer()
        self.status_update_timer.timeout.connect(self.update_detection_status)
        self.status_update_timer.start(1000)  # Update every second
        
        # Setup UI
        self.setup_ui()
        self.setup_menu()
        self.setup_toolbar()
        self.setup_statusbar()
        self.connect_signals()
        self.apply_theme()
        
        # Load saved window geometry
        self.load_window_geometry()
        
        logger.info("Face Attendance System initialized")
        
    def setup_ui(self):
        """Setup the main user interface"""
        self.setWindowTitle("Face Attendance System v3.0")
        self.setMinimumSize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Camera and controls section
        camera_frame = QFrame()
        camera_frame.setFrameStyle(QFrame.Box)
        camera_layout = QVBoxLayout(camera_frame)
        
        # Camera view label
        self.camera_label = QLabel("Camera View")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
        camera_layout.addWidget(self.camera_label)
        
        # Camera controls
        camera_controls = QHBoxLayout()
        
        self.start_button = QPushButton("Start Camera")
        self.start_button.setMinimumHeight(40)
        camera_controls.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setEnabled(False)
        camera_controls.addWidget(self.stop_button)
        
        self.reset_session_button = QPushButton("Reset Session")
        self.reset_session_button.setMinimumHeight(40)
        camera_controls.addWidget(self.reset_session_button)
        
        # Register Unknown button
        self.register_unknown_button = QPushButton("Register Unknown")
        self.register_unknown_button.setMinimumHeight(40)
        self.register_unknown_button.setEnabled(False)  # Initially disabled
        camera_controls.addWidget(self.register_unknown_button)
        
        camera_layout.addLayout(camera_controls)
        main_layout.addWidget(camera_frame)
        
        # Splitter for thumbnails and attendance log
        splitter = QSplitter(Qt.Horizontal)
        
        # Thumbnails panel
        self.thumbnail_panel = ThumbnailPanel()
        splitter.addWidget(self.thumbnail_panel)
        
        # Attendance log panel
        log_frame = QFrame()
        log_frame.setFrameStyle(QFrame.Box)
        log_layout = QVBoxLayout(log_frame)
        
        log_title = QLabel("Attendance Log")
        log_title.setAlignment(Qt.AlignCenter)
        log_title.setFont(QFont("Arial", 12, QFont.Bold))
        log_layout.addWidget(log_title)
        
        self.attendance_text = QTextEdit()
        self.attendance_text.setMaximumHeight(200)
        self.attendance_text.setReadOnly(True)
        log_layout.addWidget(self.attendance_text)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        self.export_button = QPushButton("Export Session")
        log_controls.addWidget(self.export_button)
        
        self.export_master_button = QPushButton("Export Master Log")
        log_controls.addWidget(self.export_master_button)
        
        log_layout.addLayout(log_controls)
        splitter.addWidget(log_frame)
        
        # Set splitter proportions
        splitter.setSizes([400, 300])
        main_layout.addWidget(splitter)
        
        # Statistics bar
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.Box)
        stats_layout = QHBoxLayout(stats_frame)
        
        self.known_label = QLabel("Known: 0")
        self.known_label.setFont(QFont("Arial", 10, QFont.Bold))
        stats_layout.addWidget(self.known_label)
        
        self.unknown_label = QLabel("Unknown: 0")
        self.unknown_label.setFont(QFont("Arial", 10, QFont.Bold))
        stats_layout.addWidget(self.unknown_label)
        
        self.total_label = QLabel("Total: 0")
        self.total_label.setFont(QFont("Arial", 10, QFont.Bold))
        stats_layout.addWidget(self.total_label)
        
        stats_layout.addStretch()
        
        self.last_detection_label = QLabel("Last Detection: None")
        self.last_detection_label.setFont(QFont("Arial", 9))
        stats_layout.addWidget(self.last_detection_label)
        
        # Detection status label
        self.detection_status_label = QLabel("Detection: Idle")
        self.detection_status_label.setFont(QFont("Arial", 9))
        self.detection_status_label.setStyleSheet("color: #666666;")
        stats_layout.addWidget(self.detection_status_label)
        
        main_layout.addWidget(stats_frame)
        
    def setup_menu(self):
        """Setup the menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        refresh_faces_action = QAction("Refresh Faces", self)
        refresh_faces_action.triggered.connect(self.refresh_faces)
        tools_menu.addAction(refresh_faces_action)
        
        cleanup_action = QAction("Cleanup Memory", self)
        cleanup_action.triggered.connect(self.cleanup_memory)
        tools_menu.addAction(cleanup_action)
        
        backup_action = QAction("Backup Master Log", self)
        backup_action.triggered.connect(self.backup_master_log)
        tools_menu.addAction(backup_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def setup_toolbar(self):
        """Setup the toolbar"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Theme toggle
        self.theme_action = QAction("🌙", self)
        self.theme_action.setToolTip("Toggle Theme")
        self.theme_action.triggered.connect(self.toggle_theme)
        toolbar.addAction(self.theme_action)
        
        toolbar.addSeparator()
        
        # Sound toggle
        self.sound_action = QAction("🔊", self)
        self.sound_action.setToolTip("Toggle Sound")
        self.sound_action.triggered.connect(self.toggle_sound)
        toolbar.addAction(self.sound_action)
        
        toolbar.addSeparator()
        
        # Settings
        settings_action = QAction("⚙️", self)
        settings_action.setToolTip("Settings")
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)
        
    def setup_statusbar(self):
        """Setup the status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Progress bar for operations
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.statusbar.addPermanentWidget(self.progress_bar)
        
        # Performance indicator
        self.performance_label = QLabel("Performance: Normal")
        self.performance_label.setStyleSheet("color: #666666; font-size: 9px;")
        self.statusbar.addPermanentWidget(self.performance_label)
        
        # Status message
        self.statusbar.showMessage("Ready")
        
    def connect_signals(self):
        """Connect all signals and slots"""
        # Button connections
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.reset_session_button.clicked.connect(self.reset_session)
        self.register_unknown_button.clicked.connect(self.register_latest_unknown)
        self.export_button.clicked.connect(self.export_session)
        self.export_master_button.clicked.connect(self.export_master_log)
        
        # Thumbnail panel connections
        self.thumbnail_panel.register_unknown_requested.connect(self.handle_register_unknown)
        
        # Configuration connections
        self.config_manager.config_updated.connect(self.on_config_updated)
        
        # Audio connections
        self.audio_manager.audio_error.connect(self.on_audio_error)
        
        # Logger connections
        self.attendance_logger.log_updated.connect(self.on_log_updated)
        self.attendance_logger.log_error.connect(self.on_log_error)
        
    def handle_register_unknown(self, detection_data):
        """Handle registration request for unknown face from thumbnail"""
        try:
            from thumbnail_widget import NameInputDialog
            
            # Show name input dialog
            dialog = NameInputDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                name = dialog.get_name()
                
                if not name:
                    QMessageBox.warning(self, "Invalid Name", "Please enter a valid name.")
                    return
                    
                # Register the face
                face_crop = detection_data.get('face_crop')
                detection_hash = detection_data.get('detection_hash', '')
                
                if face_crop is None:
                    QMessageBox.critical(self, "Error", "No face image available for registration.")
                    return
                    
                # Register the face
                success = self.face_manager.register_unknown_face(face_crop, detection_hash, name)
                
                if success:
                    # Update detection worker with new encodings
                    if self.detection_worker and self.detection_worker.running:
                        encodings = self.face_manager.get_face_encodings()
                        names = self.face_manager.get_face_names()
                        self.detection_worker.update_known_faces(encodings, names)
                    
                    # Remove the unknown face thumbnail
                    self.thumbnail_panel.remove_unknown_face(detection_hash)
                    
                    # Update attendance log with the new known person
                    timestamp = detection_data.get('timestamp', datetime.now())
                    self.attendance_logger.append_to_master_log(name, "Present", timestamp)
                    
                    # Add to session records
                    self.session_records.append({
                        'Name': name,
                        'Date': timestamp.strftime('%Y-%m-%d'),
                        'Time': timestamp.strftime('%H:%M:%S'),
                        'Status': 'Present'
                    })
                    
                    # Update attendance display
                    self._update_attendance_display({
                        'name': name,
                        'status': 'Present',
                        'timestamp': timestamp
                    })
                    
                    # Update statistics
                    self._update_statistics()
                    
                    # Play success sound
                    self.audio_manager.play_success_sound()
                    
                    # Show success message
                    QMessageBox.information(self, "Success", f"✅ {name} registered successfully!")
                    
                    # Update last detection label
                    self.last_detection_label.setText(f"Last Detection: {name} ({timestamp.strftime('%H:%M:%S')})")
                    
                    logger.info(f"Successfully registered unknown face as: {name}")
                    self.statusbar.showMessage(f"Registered: {name}")
                    
                else:
                    QMessageBox.critical(self, "Registration Failed", "Failed to register the face. Please try again.")
                    
        except Exception as e:
            logger.error(f"Error handling registration: {e}")
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")
            
    def register_latest_unknown(self):
        """Handle register latest unknown button click"""
        try:
            if self.latest_unknown_frame is None or self.latest_unknown_detection is None:
                QMessageBox.warning(self, "No Unknown Face", 
                                  "No unknown face has been detected yet.\n"
                                  "Please wait for an unknown face to be detected first.")
                return
                
            # Check if the face is still available (not already registered)
            if self.latest_unknown_detection.get('face_crop') is None:
                QMessageBox.warning(self, "Face Not Available", 
                                  "The unknown face is no longer available for registration.\n"
                                  "Please wait for a new unknown face to be detected.")
                return
                
            logger.info(f"Starting registration for latest unknown face. Hash: {self.latest_unknown_detection.get('detection_hash', 'Unknown')}")
                
            from thumbnail_widget import NameInputDialog
            
            # Show name input dialog
            dialog = NameInputDialog(self)
            if dialog.exec_() == QDialog.Accepted:
                name = dialog.get_name()
                
                if not name:
                    QMessageBox.warning(self, "Invalid Name", "Please enter a valid name.")
                    return
                    
                # Register the face using the latest unknown detection
                face_crop = self.latest_unknown_detection.get('face_crop')
                detection_hash = self.latest_unknown_detection.get('detection_hash', '')
                
                if face_crop is None:
                    QMessageBox.critical(self, "Error", "No face image available for registration.")
                    return
                    
                # Register the face
                success = self.face_manager.register_unknown_face(face_crop, detection_hash, name)
                
                if success:
                    # Update detection worker with new encodings
                    if self.detection_worker and self.detection_worker.running:
                        encodings = self.face_manager.get_face_encodings()
                        names = self.face_manager.get_face_names()
                        self.detection_worker.update_known_faces(encodings, names)
                    
                    # Remove the unknown face thumbnail if it exists
                    self.thumbnail_panel.remove_unknown_face(detection_hash)
                    
                    # Update attendance log with the new known person
                    timestamp = self.latest_unknown_detection.get('timestamp', datetime.now())
                    self.attendance_logger.append_to_master_log(name, "Present", timestamp)
                    
                    # Add to session records
                    self.session_records.append({
                        'Name': name,
                        'Date': timestamp.strftime('%Y-%m-%d'),
                        'Time': timestamp.strftime('%H:%M:%S'),
                        'Status': 'Present'
                    })
                    
                    # Update attendance display
                    self._update_attendance_display({
                        'name': name,
                        'status': 'Present',
                        'timestamp': timestamp
                    })
                    
                    # Update statistics
                    self._update_statistics()
                    
                    # Play success sound
                    self.audio_manager.play_success_sound()
                    
                    # Show success message
                    QMessageBox.information(self, "Success", f"✅ {name} registered successfully!")
                    
                    # Update last detection label
                    self.last_detection_label.setText(f"Last Detection: {name} ({timestamp.strftime('%H:%M:%S')})")
                    
                    # Clear the latest unknown frame tracking
                    self.latest_unknown_frame = None
                    self.latest_unknown_detection = None
                    
                    # Disable the register button
                    self.register_unknown_button.setEnabled(False)
                    
                    logger.info(f"Successfully registered latest unknown face as: {name}")
                    self.statusbar.showMessage(f"Registered: {name}")
                    
                else:
                    QMessageBox.critical(self, "Registration Failed", "Failed to register the face. Please try again.")
                    
        except Exception as e:
            logger.error(f"Error registering latest unknown: {e}")
            QMessageBox.critical(self, "Error", f"Registration failed: {str(e)}")
            
    def apply_theme(self):
        """Apply the current theme"""
        theme = self.config_manager.get('theme', 'light')
        stylesheet = self.config_manager.get_theme_stylesheet(theme)
        self.setStyleSheet(stylesheet)
        
        # Update theme button icon
        if theme == 'dark':
            self.theme_action.setText("☀️")
            self.theme_action.setToolTip("Switch to Light Theme")
        else:
            self.theme_action.setText("🌙")
            self.theme_action.setToolTip("Switch to Dark Theme")
            
    def load_window_geometry(self):
        """Load saved window geometry"""
        geometry = self.config_manager.get_window_geometry()
        self.resize(geometry['width'], geometry['height'])
        self.move(geometry['x'], geometry['y'])
        
    def start_camera(self):
        """Start camera capture and detection"""
        try:
            if self.is_running:
                return
                
            # Get configuration
            camera_index = self.config_manager.get('camera_index', 0)
            frame_skip = self.config_manager.get('frame_skip', 1)
            
            # Initialize camera worker
            self.camera_worker = CameraWorker(camera_index)
            self.camera_worker.set_frame_skip(frame_skip)
            self.camera_worker.frame_ready.connect(self.on_frame_ready)
            self.camera_worker.camera_error.connect(self.on_camera_error)
            self.camera_worker.camera_status.connect(self.on_camera_status)
            
            # Initialize detection worker
            encodings = self.face_manager.get_face_encodings()
            names = self.face_manager.get_face_names()
            self.detection_worker = DetectionWorker(encodings, names)
            
            # Configure detection interval from settings
            detection_interval = self.config_manager.get('detection_interval', 1.5)
            self.detection_worker.set_processing_interval(detection_interval)
            
            self.detection_worker.detection_result.connect(self.on_detection_result)
            self.detection_worker.detection_error.connect(self.on_detection_error)
            
            # Start workers
            self.camera_worker.start()
            self.detection_worker.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.register_unknown_button.setEnabled(False)  # Will be enabled when unknown face detected
            self.is_running = True
            
            # Start status update timer
            self.status_update_timer.start(1000)
            
            self.statusbar.showMessage("Camera started")
            logger.info("Camera and detection started")
            
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        """Stop camera capture and detection"""
        try:
            if not self.is_running:
                return
                
            # Stop workers
            if self.camera_worker:
                self.camera_worker.stop()
                self.camera_worker = None
                
            if self.detection_worker:
                self.detection_worker.stop()
                self.detection_worker = None
                
            # Update UI
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.is_running = False
            
            # Clear camera view
            self.camera_label.setText("Camera Stopped")
            self.camera_label.setStyleSheet("border: 2px solid #cccccc; background-color: #f0f0f0;")
            
            # Clear latest unknown frame tracking
            self.latest_unknown_frame = None
            self.latest_unknown_detection = None
            
            # Disable register unknown button
            self.register_unknown_button.setEnabled(False)
            
            # Stop status update timer
            self.status_update_timer.stop()
            
            self.statusbar.showMessage("Camera stopped")
            logger.info("Camera and detection stopped")
            
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
            
    def on_frame_ready(self, frame):
        """Handle frame from camera worker"""
        try:
            # Convert frame to QPixmap for display (always update display for smooth video)
            pixmap = self._cv2_to_qpixmap(frame)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            
            # Queue frame for detection processing (instead of immediate processing)
            if self.detection_worker and self.detection_worker.running:
                self.detection_worker.queue_frame(frame)
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            
    def on_detection_result(self, result):
        """Handle detection results from detection worker"""
        try:
            timestamp = result.get('timestamp', datetime.now())
            detections = result.get('detections', [])
            
            for detection in detections:
                self._process_detection(detection, timestamp)
                
            # Update statistics
            self._update_statistics()
            
        except Exception as e:
            logger.error(f"Error processing detection result: {e}")
            
    def _process_detection(self, detection, timestamp):
        """Process a single detection"""
        try:
            name = detection.get('name', 'Unknown')
            detection_type = detection.get('type', 'unknown')
            face_crop = detection.get('face_crop')
            detection_hash = detection.get('detection_hash', '')
            
            # Add timestamp to detection data
            detection['timestamp'] = timestamp
            
            # Log to master attendance log
            status = detection.get('status', 'Unknown')
            self.attendance_logger.append_to_master_log(name, status, timestamp)
            
            # Save unknown face if needed
            if detection_type == 'unknown' and face_crop is not None:
                self.face_manager.save_unknown_face(face_crop, detection_hash)
                
                # Track the latest unknown frame for registration
                self.latest_unknown_frame = face_crop.copy()
                self.latest_unknown_detection = detection.copy()
                
                # Enable the register unknown button
                self.register_unknown_button.setEnabled(True)
                
                logger.info(f"Latest unknown face tracked for registration. Hash: {detection_hash}")
                
            # Add to session records
            self.session_records.append({
                'Name': name,
                'Date': timestamp.strftime('%Y-%m-%d'),
                'Time': timestamp.strftime('%H:%M:%S'),
                'Status': status
            })
            
            # Add thumbnail
            if detection_type == 'known':
                self.thumbnail_panel.add_known_face(detection)
            else:
                self.thumbnail_panel.add_unknown_face(detection)
                
            # Play sound notification
            if detection_type == 'known':
                self.audio_manager.play_known_face_sound()
            else:
                self.audio_manager.play_unknown_face_sound()
                
            # Update attendance display
            self._update_attendance_display(detection)
            
            # Update last detection label
            self.last_detection_label.setText(f"Last Detection: {name} ({timestamp.strftime('%H:%M:%S')})")
            
            logger.info(f"Processed detection: {name} ({status})")
            
        except Exception as e:
            logger.error(f"Error processing detection: {e}")
            
    def _update_attendance_display(self, detection):
        """Update the attendance display"""
        try:
            name = detection.get('name', 'Unknown')
            status = detection.get('status', 'Unknown')
            timestamp = detection.get('timestamp', datetime.now())
            
            # Format the entry
            time_str = timestamp.strftime('%H:%M:%S')
            status_icon = "✓" if status == "Present" else "?"
            
            entry = f"[{time_str}] {status_icon} {name} ({status})\n"
            
            # Add to text display
            current_text = self.attendance_text.toPlainText()
            self.attendance_text.setText(entry + current_text)
            
        except Exception as e:
            logger.error(f"Error updating attendance display: {e}")
            
    def _update_statistics(self):
        """Update statistics display"""
        try:
            summary = self.thumbnail_panel.get_session_summary()
            
            self.known_label.setText(f"Known: {summary['known_faces']}")
            self.unknown_label.setText(f"Unknown: {summary['unknown_faces']}")
            self.total_label.setText(f"Total: {summary['total_faces']}")
            
        except Exception as e:
            logger.error(f"Error updating statistics: {e}")
            
    def update_detection_status(self):
        """Update detection status display"""
        try:
            if not self.detection_worker or not self.detection_worker.running:
                self.detection_status_label.setText("Detection: Stopped")
                self.detection_status_label.setStyleSheet("color: #999999;")
                self.performance_label.setText("Performance: Idle")
                self.performance_label.setStyleSheet("color: #999999; font-size: 9px;")
                return
                
            # Get processing status from detection worker
            status = self.detection_worker.get_processing_status()
            
            if status['has_pending_frame']:
                time_since = status['time_since_last_processing']
                interval = status['processing_interval']
                remaining = max(0, interval - time_since)
                
                if remaining > 0:
                    self.detection_status_label.setText(f"Detection: Processing in {remaining:.1f}s")
                    self.detection_status_label.setStyleSheet("color: #ff8c00;")  # Orange
                    self.performance_label.setText("Performance: Good")
                    self.performance_label.setStyleSheet("color: #00aa00; font-size: 9px;")
                else:
                    self.detection_status_label.setText("Detection: Processing...")
                    self.detection_status_label.setStyleSheet("color: #ff0000;")  # Red
                    self.performance_label.setText("Performance: Processing")
                    self.performance_label.setStyleSheet("color: #ff8c00; font-size: 9px;")
            else:
                self.detection_status_label.setText("Detection: Waiting for frame")
                self.detection_status_label.setStyleSheet("color: #666666;")  # Gray
                self.performance_label.setText("Performance: Normal")
                self.performance_label.setStyleSheet("color: #666666; font-size: 9px;")
                
        except Exception as e:
            logger.error(f"Error updating detection status: {e}")
            self.detection_status_label.setText("Detection: Error")
            self.detection_status_label.setStyleSheet("color: #ff0000;")
            self.performance_label.setText("Performance: Error")
            self.performance_label.setStyleSheet("color: #ff0000; font-size: 9px;")
            
    def reset_session(self):
        """Reset the current session"""
        try:
            reply = QMessageBox.question(
                self, "Reset Session",
                "Are you sure you want to reset the current session?\n"
                "This will clear all thumbnails and session data, but won't affect the master log.",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Clear thumbnails
                self.thumbnail_panel.clear_session()
                
                # Clear session records
                self.session_records.clear()
                
                # Reset detection worker session
                if self.detection_worker:
                    self.detection_worker.reset_session()
                    
                # Clear attendance display
                self.attendance_text.clear()
                
                # Update statistics
                self._update_statistics()
                
                # Update last detection
                self.last_detection_label.setText("Last Detection: None")
                
                # Clear latest unknown frame tracking
                self.latest_unknown_frame = None
                self.latest_unknown_detection = None
                
                # Disable register unknown button
                self.register_unknown_button.setEnabled(False)
                
                self.statusbar.showMessage("Session reset")
                logger.info("Session reset completed")
                
        except Exception as e:
            logger.error(f"Error resetting session: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset session: {str(e)}")
            
    def export_session(self):
        """Export current session to Excel"""
        try:
            if not self.session_records:
                QMessageBox.information(self, "Export", "No session records to export")
                return
                
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Session", "session_attendance.xlsx",
                "Excel Files (*.xlsx)"
            )
            
            if filename:
                success = self.attendance_logger.export_session_summary(
                    self.session_records, filename
                )
                
                if success:
                    QMessageBox.information(self, "Export", f"Session exported to {filename}")
                    self.statusbar.showMessage(f"Session exported to {filename}")
                else:
                    QMessageBox.critical(self, "Export Error", "Failed to export session")
                    
        except Exception as e:
            logger.error(f"Error exporting session: {e}")
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            
    def export_master_log(self):
        """Export master attendance log"""
        try:
            master_file = self.attendance_logger.master_file
            
            if not os.path.exists(master_file):
                QMessageBox.information(self, "Export", "No master log file exists yet")
                return
                
            # Get backup directory
            backup_dir = "backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"attendance_log_backup_{timestamp}.xlsx"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            import shutil
            shutil.copy2(master_file, backup_path)
            
            QMessageBox.information(
                self, "Export", 
                f"Master log backed up to {backup_path}\n"
                f"Original file: {master_file}"
            )
            
            self.statusbar.showMessage(f"Master log backed up to {backup_path}")
            
        except Exception as e:
            logger.error(f"Error exporting master log: {e}")
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            
    def backup_master_log(self):
        """Backup the master attendance log"""
        try:
            success = self.attendance_logger.backup_master_log()
            
            if success:
                QMessageBox.information(self, "Backup", "Master log backed up successfully")
                self.statusbar.showMessage("Master log backed up")
            else:
                QMessageBox.warning(self, "Backup", "No master log file to backup")
                
        except Exception as e:
            logger.error(f"Error backing up master log: {e}")
            QMessageBox.critical(self, "Error", f"Backup failed: {str(e)}")
            
    def refresh_faces(self):
        """Refresh known faces from directory"""
        try:
            self.face_manager.refresh_faces()
            
            # Update detection worker if running
            if self.detection_worker and self.detection_worker.running:
                encodings = self.face_manager.get_face_encodings()
                names = self.face_manager.get_face_names()
                self.detection_worker.update_known_faces(encodings, names)
                
            QMessageBox.information(self, "Refresh", "Faces refreshed successfully")
            self.statusbar.showMessage("Faces refreshed")
            logger.info("Faces refreshed from directory")
            
        except Exception as e:
            logger.error(f"Error refreshing faces: {e}")
            QMessageBox.critical(self, "Error", f"Failed to refresh faces: {str(e)}")
            
    def cleanup_memory(self):
        """Clean up memory and optimize performance"""
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            
            # Clear any cached frames
            if hasattr(self, 'latest_unknown_frame') and self.latest_unknown_frame is not None:
                del self.latest_unknown_frame
                self.latest_unknown_frame = None
                
            if hasattr(self, 'latest_unknown_detection') and self.latest_unknown_detection is not None:
                del self.latest_unknown_detection
                self.latest_unknown_detection = None
                
            # Clear thumbnail panel memory if possible
            if hasattr(self, 'thumbnail_panel'):
                self.thumbnail_panel.clear_session()
                
            QMessageBox.information(self, "Memory Cleanup", 
                                  f"Memory cleanup completed. Garbage collected: {collected} objects")
            self.statusbar.showMessage("Memory cleanup completed")
            logger.info(f"Memory cleanup completed. Garbage collected: {collected} objects")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            QMessageBox.critical(self, "Error", f"Memory cleanup failed: {str(e)}")
            
    def show_settings(self):
        """Show settings dialog"""
        try:
            dialog = SettingsDialog(self.config_manager, self)
            if dialog.exec_() == QDialog.Accepted:
                # Apply new settings
                self.apply_theme()
                
                # Update audio manager
                sound_enabled = self.config_manager.get('sound_enabled', True)
                self.audio_manager.set_sound_enabled(sound_enabled)
                
                # Update thumbnail size
                thumbnail_size = self.config_manager.get('thumbnail_size', 80)
                self.thumbnail_panel.known_thumbnails.set_thumbnail_size(thumbnail_size)
                self.thumbnail_panel.unknown_thumbnails.set_thumbnail_size(thumbnail_size)
                
                self.statusbar.showMessage("Settings applied")
                
        except Exception as e:
            logger.error(f"Error showing settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show settings: {str(e)}")
            
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        try:
            new_theme = self.config_manager.toggle_theme()
            self.apply_theme()
            
            self.statusbar.showMessage(f"Theme switched to {new_theme}")
            logger.info(f"Theme switched to {new_theme}")
            
        except Exception as e:
            logger.error(f"Error toggling theme: {e}")
            
    def toggle_sound(self):
        """Toggle sound on/off"""
        try:
            new_sound = self.config_manager.toggle_sound()
            self.audio_manager.set_sound_enabled(new_sound)
            
            # Update button icon
            if new_sound:
                self.sound_action.setText("🔊")
                self.sound_action.setToolTip("Disable Sound")
            else:
                self.sound_action.setText("🔇")
                self.sound_action.setToolTip("Enable Sound")
                
            self.statusbar.showMessage(f"Sound {'enabled' if new_sound else 'disabled'}")
            logger.info(f"Sound {'enabled' if new_sound else 'disabled'}")
            
        except Exception as e:
            logger.error(f"Error toggling sound: {e}")
            
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Face Attendance System",
            "Face Attendance System v3.0\n\n"
            "A robust face recognition system for attendance tracking.\n\n"
            "Features:\n"
            "• Multi-threaded camera capture\n"
            "• Face detection and recognition\n"
            "• Session-based attendance tracking\n"
            "• Master Excel logging\n"
            "• Thumbnail display\n"
            "• Audio notifications\n"
            "• Theme switching\n\n"
            "Built with PyQt5, OpenCV, and face_recognition."
        )
        
    def _cv2_to_qpixmap(self, cv_img):
        """Convert OpenCV image to QPixmap with memory optimization"""
        try:
            if cv_img is None:
                return QPixmap()
                
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            # Clear intermediate variables to free memory
            del rgb_image, q_image
            
            return pixmap
            
        except Exception as e:
            logger.error(f"Error converting image to pixmap: {e}")
            return QPixmap()
            
    def on_camera_error(self, error_msg):
        """Handle camera errors"""
        logger.error(f"Camera error: {error_msg}")
        self.statusbar.showMessage(f"Camera error: {error_msg}")
        QMessageBox.warning(self, "Camera Error", error_msg)
        
    def on_camera_status(self, status_msg):
        """Handle camera status updates"""
        self.statusbar.showMessage(status_msg)
        logger.info(f"Camera status: {status_msg}")
        
    def on_detection_error(self, error_msg):
        """Handle detection errors"""
        logger.error(f"Detection error: {error_msg}")
        self.statusbar.showMessage(f"Detection error: {error_msg}")
        
    def on_log_updated(self, message):
        """Handle log update messages"""
        self.statusbar.showMessage(message)
        logger.info(f"Log update: {message}")
        
    def on_log_error(self, error_msg):
        """Handle log error messages"""
        logger.error(f"Log error: {error_msg}")
        self.statusbar.showMessage(f"Log error: {error_msg}")
        QMessageBox.warning(self, "Log Error", error_msg)
        
    def on_audio_error(self, error_msg):
        """Handle audio error messages"""
        logger.error(f"Audio error: {error_msg}")
        self.statusbar.showMessage(f"Audio error: {error_msg}")
        
    def on_config_updated(self, config):
        """Handle configuration updates"""
        logger.info("Configuration updated")
        
        # Update detection interval if detection worker is running
        if self.detection_worker and self.detection_worker.running:
            detection_interval = config.get('detection_interval', 1.5)
            self.detection_worker.set_processing_interval(detection_interval)
            logger.info(f"Detection interval updated to {detection_interval} seconds")
        
    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Stop camera if running
            if self.is_running:
                self.stop_camera()
                
            # Stop status update timer
            self.status_update_timer.stop()
            
            # Clear memory references
            self.latest_unknown_frame = None
            self.latest_unknown_detection = None
            
            # Save window geometry
            geometry = self.geometry()
            self.config_manager.update_window_geometry(
                geometry.x(), geometry.y(),
                geometry.width(), geometry.height()
            )
            
            # Cleanup audio manager
            self.audio_manager.cleanup()
            
            logger.info("Application closing")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
            event.accept()

def main():
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Face Attendance System")
        app.setApplicationVersion("3.0")
        
        # Create and show main window
        window = AttendanceApp()
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application startup error: {e}")
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
