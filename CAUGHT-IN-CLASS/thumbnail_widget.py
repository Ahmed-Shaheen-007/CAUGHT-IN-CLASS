import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QScrollArea, QFrame, QPushButton, QSizePolicy, QDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ThumbnailWidget(QWidget):
    """Widget for displaying face thumbnails"""
    
    thumbnail_clicked = pyqtSignal(dict)  # Emit when thumbnail is clicked
    register_unknown_requested = pyqtSignal(dict)  # Emit when registration is requested
    
    def __init__(self, title="Thumbnails", parent=None, show_register_button=False):
        super().__init__(parent)
        self.title = title
        self.thumbnails = {}  # Store thumbnail data
        self.thumbnail_size = 80
        self.show_register_button = show_register_button  # Whether to show register button
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Title label
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        # Scroll area for thumbnails
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setMaximumHeight(120)
        
        # Container widget for thumbnails
        self.thumbnail_container = QWidget()
        self.thumbnail_layout = QHBoxLayout(self.thumbnail_container)
        self.thumbnail_layout.setAlignment(Qt.AlignLeft)
        self.thumbnail_layout.setSpacing(10)
        self.thumbnail_layout.setContentsMargins(10, 5, 10, 5)
        
        self.scroll_area.setWidget(self.thumbnail_container)
        layout.addWidget(self.scroll_area)
        
    def add_thumbnail(self, detection_data):
        """Add a new thumbnail for face detection"""
        try:
            name = detection_data.get('name', 'Unknown')
            face_crop = detection_data.get('face_crop')
            detection_hash = detection_data.get('detection_hash', '')
            timestamp = detection_data.get('timestamp', datetime.now())
            
            if face_crop is None:
                logger.warning(f"No face crop data for {name}")
                return
                
            # Create thumbnail widget
            thumbnail_widget = self._create_thumbnail_widget(
                name, face_crop, detection_hash, timestamp
            )
            
            # Store thumbnail data
            self.thumbnails[detection_hash] = {
                'widget': thumbnail_widget,
                'data': detection_data
            }
            
            # Add to layout
            self.thumbnail_layout.addWidget(thumbnail_widget)
            
            logger.debug(f"Added thumbnail for {name}")
            
        except Exception as e:
            logger.error(f"Error adding thumbnail: {e}")
            
    def _create_thumbnail_widget(self, name, face_crop, detection_hash, timestamp):
        """Create a single thumbnail widget"""
        # Create container frame
        frame = QFrame()
        frame.setFrameStyle(QFrame.Box)
        frame.setLineWidth(2)
        frame.setMidLineWidth(1)
        
        # Adjust size based on whether we need to show register button
        if self.show_register_button and name == 'Unknown':
            frame.setMaximumSize(self.thumbnail_size + 20, self.thumbnail_size + 80)
            frame.setMinimumSize(self.thumbnail_size + 20, self.thumbnail_size + 80)
        else:
            frame.setMaximumSize(self.thumbnail_size + 20, self.thumbnail_size + 40)
            frame.setMinimumSize(self.thumbnail_size + 20, self.thumbnail_size + 40)
        
        # Layout for thumbnail
        layout = QVBoxLayout(frame)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Convert face crop to QPixmap
        pixmap = self._cv2_to_qpixmap(face_crop, self.thumbnail_size)
        
        # Image label
        image_label = QLabel()
        image_label.setPixmap(pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(self.thumbnail_size, self.thumbnail_size)
        image_label.setMaximumSize(self.thumbnail_size, self.thumbnail_size)
        layout.addWidget(image_label)
        
        # Name label
        name_label = QLabel(name)
        name_label.setAlignment(Qt.AlignCenter)
        name_label.setFont(QFont("Arial", 8))
        name_label.setWordWrap(True)
        name_label.setMaximumHeight(20)
        layout.addWidget(name_label)
        
        # Add register button for unknown faces
        if self.show_register_button and name == 'Unknown':
            register_button = QPushButton("Register as Known")
            register_button.setMaximumHeight(25)
            register_button.setFont(QFont("Arial", 7))
            register_button.clicked.connect(lambda: self._on_register_clicked(detection_hash))
            layout.addWidget(register_button)
        
        # Make frame clickable (but not for unknown faces with register button)
        if not (self.show_register_button and name == 'Unknown'):
            frame.mousePressEvent = lambda event: self._on_thumbnail_clicked(detection_hash)
            frame.setCursor(Qt.PointingHandCursor)
        
        return frame
        
    def _on_register_clicked(self, detection_hash):
        """Handle register button click"""
        if detection_hash in self.thumbnails:
            data = self.thumbnails[detection_hash]['data']
            self.register_unknown_requested.emit(data)
            logger.debug(f"Register unknown requested for hash: {detection_hash}")
        
    def _cv2_to_qpixmap(self, cv_img, size):
        """Convert OpenCV image to QPixmap"""
        try:
            if cv_img is None:
                # Create a placeholder image
                placeholder = np.zeros((size, size, 3), dtype=np.uint8)
                placeholder[:] = (128, 128, 128)  # Gray color
                cv_img = placeholder
                
            # Resize image
            resized = cv2.resize(cv_img, (size, size))
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            return pixmap
            
        except Exception as e:
            logger.error(f"Error converting image to pixmap: {e}")
            # Return a placeholder pixmap
            placeholder_pixmap = QPixmap(size, size)
            placeholder_pixmap.fill(QColor(128, 128, 128))
            return placeholder_pixmap
            
    def _on_thumbnail_clicked(self, detection_hash):
        """Handle thumbnail click event"""
        if detection_hash in self.thumbnails:
            data = self.thumbnails[detection_hash]['data']
            self.thumbnail_clicked.emit(data)
            
    def remove_thumbnail(self, detection_hash):
        """Remove a specific thumbnail"""
        if detection_hash in self.thumbnails:
            widget = self.thumbnails[detection_hash]['widget']
            self.thumbnail_layout.removeWidget(widget)
            widget.deleteLater()
            del self.thumbnails[detection_hash]
            logger.debug(f"Removed thumbnail: {detection_hash}")
            
    def clear_all_thumbnails(self):
        """Clear all thumbnails"""
        for detection_hash in list(self.thumbnails.keys()):
            self.remove_thumbnail(detection_hash)
        logger.info("All thumbnails cleared")
        
    def get_thumbnail_count(self):
        """Get the number of thumbnails"""
        return len(self.thumbnails)
        
    def set_thumbnail_size(self, size):
        """Set thumbnail size"""
        self.thumbnail_size = size
        # Refresh all thumbnails
        self._refresh_thumbnails()
        
    def _refresh_thumbnails(self):
        """Refresh all thumbnails with new size"""
        # Store current data
        current_data = {}
        for detection_hash, thumbnail_info in self.thumbnails.items():
            current_data[detection_hash] = thumbnail_info['data']
            
        # Clear and recreate
        self.clear_all_thumbnails()
        
        # Recreate with new size
        for detection_hash, data in current_data.items():
            self.add_thumbnail(data)
            
    def get_thumbnail_data(self):
        """Get all thumbnail data"""
        return {hash_val: info['data'] for hash_val, info in self.thumbnails.items()}
        
    def update_thumbnail(self, detection_hash, new_data):
        """Update an existing thumbnail"""
        if detection_hash in self.thumbnails:
            # Remove old thumbnail
            self.remove_thumbnail(detection_hash)
            # Add new one
            self.add_thumbnail(new_data)
            
    def set_title(self, title):
        """Set the title of the thumbnail widget"""
        self.title = title
        # Update the title label
        title_label = self.findChild(QLabel)
        if title_label:
            title_label.setText(title)

class ThumbnailPanel(QWidget):
    """Panel containing multiple thumbnail widgets"""
    
    register_unknown_requested = pyqtSignal(dict)  # Forward the signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        
        # Known faces thumbnails
        self.known_thumbnails = ThumbnailWidget("Known Faces", show_register_button=False)
        layout.addWidget(self.known_thumbnails)
        
        # Unknown faces thumbnails
        self.unknown_thumbnails = ThumbnailWidget("Unknown Faces", show_register_button=True)
        layout.addWidget(self.unknown_thumbnails)
        
        # Connect signals
        self.known_thumbnails.thumbnail_clicked.connect(self._on_known_thumbnail_clicked)
        self.unknown_thumbnails.thumbnail_clicked.connect(self._on_unknown_thumbnail_clicked)
        self.unknown_thumbnails.register_unknown_requested.connect(self.register_unknown_requested.emit)
        
    def add_known_face(self, detection_data):
        """Add a known face thumbnail"""
        self.known_thumbnails.add_thumbnail(detection_data)
        
    def add_unknown_face(self, detection_data):
        """Add an unknown face thumbnail"""
        self.unknown_thumbnails.add_thumbnail(detection_data)
        
    def remove_unknown_face(self, detection_hash):
        """Remove an unknown face thumbnail (after registration)"""
        self.unknown_thumbnails.remove_thumbnail(detection_hash)
        
    def clear_session(self):
        """Clear all session thumbnails"""
        self.known_thumbnails.clear_all_thumbnails()
        self.unknown_thumbnails.clear_all_thumbnails()
        
    def get_session_summary(self):
        """Get session summary"""
        return {
            'known_faces': self.known_thumbnails.get_thumbnail_count(),
            'unknown_faces': self.unknown_thumbnails.get_thumbnail_count(),
            'total_faces': (self.known_thumbnails.get_thumbnail_count() + 
                           self.unknown_thumbnails.get_thumbnail_count())
        }
        
    def _on_known_thumbnail_clicked(self, data):
        """Handle known face thumbnail click"""
        logger.debug(f"Known face thumbnail clicked: {data.get('name', 'Unknown')}")
        # Emit signal or show details dialog
        
    def _on_unknown_thumbnail_clicked(self, data):
        """Handle unknown face thumbnail click"""
        logger.debug(f"Unknown face thumbnail clicked: {data.get('detection_hash', 'Unknown')}")
        # Emit signal or show details dialog 

class NameInputDialog(QDialog):
    """Dialog for inputting a name when registering an unknown face"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = ""
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("Register Unknown Face")
        self.setModal(True)
        self.setFixedSize(300, 150)
        
        layout = QVBoxLayout(self)
        
        # Instructions label
        instruction_label = QLabel("Enter the name for this person:")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setFont(QFont("Arial", 10))
        layout.addWidget(instruction_label)
        
        # Name input field
        from PyQt5.QtWidgets import QLineEdit
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name here...")
        self.name_input.setFont(QFont("Arial", 12))
        self.name_input.textChanged.connect(self._on_name_changed)
        layout.addWidget(self.name_input)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        button_layout.addStretch()
        
        self.register_button = QPushButton("Register")
        self.register_button.setEnabled(False)  # Initially disabled
        self.register_button.clicked.connect(self.accept)
        button_layout.addWidget(self.register_button)
        
        layout.addLayout(button_layout)
        
        # Set focus to name input
        self.name_input.setFocus()
        
    def _on_name_changed(self, text):
        """Handle name input changes"""
        self.name = text.strip()
        self.register_button.setEnabled(len(self.name) > 0)
        
    def get_name(self):
        """Get the entered name"""
        return self.name
        
    def keyPressEvent(self, event):
        """Handle key press events"""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.register_button.isEnabled():
                self.accept()
        elif event.key() == Qt.Key_Escape:
            self.reject()
        else:
            super().keyPressEvent(event) 