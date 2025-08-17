import cv2
import numpy as np
import face_recognition
import os
import logging
from PIL import Image
import hashlib
from datetime import datetime
import pickle
import json

logger = logging.getLogger(__name__)

class FaceManager:
    """Manages face loading, processing, and storage"""
    
    def __init__(self, faces_dir='faces', unknowns_dir='unknowns'):
        self.faces_dir = faces_dir
        self.unknowns_dir = unknowns_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_images = {}  # Store original images for thumbnails
        
        # Create directories if they don't exist
        os.makedirs(faces_dir, exist_ok=True)
        os.makedirs(unknowns_dir, exist_ok=True)
        
        # Load known faces
        self._load_known_faces()
        
        # Load persistent encodings if available
        self._load_persistent_encodings()
        
    def _load_persistent_encodings(self):
        """Load persistent face encodings from file"""
        try:
            encodings_file = 'encodings.pkl'
            if os.path.exists(encodings_file):
                with open(encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
                    logger.info(f"Loaded {len(self.known_face_names)} persistent encodings")
                    
                    # Also load the corresponding images for thumbnails
                    self._load_images_for_persistent_encodings()
                    
        except Exception as e:
            logger.error(f"Error loading persistent encodings: {e}")
            
    def _load_images_for_persistent_encodings(self):
        """Load images for persistent encodings"""
        try:
            for name in self.known_face_names:
                # Look for image in faces directory
                image_path = self._find_image_for_name(name)
                if image_path:
                    image = self._load_and_process_image(image_path)
                    if image is not None:
                        self.known_face_images[name] = image
                        logger.debug(f"Loaded image for persistent encoding: {name}")
                        
        except Exception as e:
            logger.error(f"Error loading images for persistent encodings: {e}")
            
    def _find_image_for_name(self, name):
        """Find image file for a given name"""
        try:
            for filename in os.listdir(self.faces_dir):
                # Check if filename starts with the name (handles timestamp format)
                if filename.startswith(name + "_") or os.path.splitext(filename)[0] == name:
                    return os.path.join(self.faces_dir, filename)
            return None
        except Exception as e:
            logger.error(f"Error finding image for name {name}: {e}")
            return None
            
    def _save_persistent_encodings(self):
        """Save face encodings to persistent file"""
        try:
            encodings_file = 'encodings.pkl'
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(encodings_file, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved {len(self.known_face_names)} encodings to persistent storage")
            return True
            
        except Exception as e:
            logger.error(f"Error saving persistent encodings: {e}")
            return False
            
    def register_unknown_face(self, face_crop, detection_hash, name):
        """Register an unknown face as a known face"""
        try:
            # Validate name
            if not name or not name.strip():
                logger.error("Invalid name provided for registration")
                return False
                
            name = name.strip()
            
            # Check if name already exists
            if name in self.known_face_names:
                logger.warning(f"Name '{name}' already exists, cannot register duplicate")
                return False
                
            # Generate face encoding
            # Ensure image is in the correct format for face_recognition
            if face_crop.dtype != np.uint8:
                face_crop = np.clip(face_crop, 0, 255).astype(np.uint8)
                
            # Convert BGR to RGB for face_recognition
            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
            # Ensure RGB image is in correct format
            if rgb_face.dtype != np.uint8:
                rgb_face = np.clip(rgb_face, 0, 255).astype(np.uint8)
                
            encodings = face_recognition.face_encodings(rgb_face)
            
            if not encodings:
                logger.error("Could not generate encoding for the face crop")
                return False
                
            face_encoding = encodings[0]
            
            # Save face image to faces directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{name}_{timestamp}.jpg"
            image_path = os.path.join(self.faces_dir, image_filename)
            
            # Save the face crop
            success = cv2.imwrite(image_path, face_crop)
            if not success:
                logger.error(f"Failed to save face image to {image_path}")
                return False
                
            # Add to known faces
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.known_face_images[name] = face_crop.copy()
            
            # Save to persistent storage
            self._save_persistent_encodings()
            
            logger.info(f"Successfully registered '{name}' as known face with image: {image_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering unknown face: {e}")
            return False
            
    def _load_known_faces(self):
        """Load known faces with robust image handling"""
        if not os.path.exists(self.faces_dir):
            logger.warning(f"Faces directory '{self.faces_dir}' not found. Creating it.")
            return
            
        loaded_count = 0
        for filename in os.listdir(self.faces_dir):
            filepath = os.path.join(self.faces_dir, filename)
            if not os.path.isfile(filepath):
                continue
                
            # Check file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}:
                logger.warning(f"Skipping unsupported format: {filename}")
                continue
                
            try:
                # Load and process image
                image = self._load_and_process_image(filepath)
                if image is None:
                    continue
                    
                # Get face encodings
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    # Extract name from filename (remove timestamp if present)
                    base_name = os.path.splitext(filename)[0]
                    if "_" in base_name:
                        # Format: name_YYYYMMDD_HHMMSS
                        name = base_name.split("_")[0]
                    else:
                        name = base_name
                    self.known_face_names.append(name)
                    self.known_face_images[name] = image.copy()  # Store for thumbnails
                    loaded_count += 1
                    logger.info(f"Loaded face: {filename} -> {name}")
                else:
                    logger.warning(f"No face found in {filename}")
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                
        logger.info(f"Successfully loaded {loaded_count} faces")
        
    def _load_and_process_image(self, image_path):
        """Load and process image with robust handling for various formats"""
        try:
            # Try OpenCV first
            image = cv2.imread(image_path)
            if image is None:
                # Try PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                # Convert RGBA to BGR if necessary
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Ensure image is uint8
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
                
            # Handle different channel counts
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # BGRA
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                elif image.shape[2] == 1:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 3:  # BGR (OpenCV default)
                    pass  # Already correct
                else:
                    logger.warning(f"Unexpected image channels: {image.shape[2]} in {image_path}")
                    return None
            elif len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                logger.warning(f"Unexpected image shape: {image.shape} in {image_path}")
                return None
                
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
            
    def save_unknown_face(self, face_crop, detection_hash):
        """Save unknown face crop to unknowns directory"""
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unknown_{timestamp}_{detection_hash[:8]}.jpg"
            filepath = os.path.join(self.unknowns_dir, filename)
            
            # Save face crop
            cv2.imwrite(filepath, face_crop)
            logger.info(f"Saved unknown face: {filename}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving unknown face: {e}")
            return None
            
    def get_face_thumbnail(self, name, size=(80, 80)):
        """Get thumbnail for a known face"""
        if name in self.known_face_images:
            image = self.known_face_images[name]
            # Resize for thumbnail
            thumbnail = cv2.resize(image, size)
            return thumbnail
        return None
        
    def refresh_faces(self):
        """Refresh known faces from directory and persistent storage"""
        # Clear current data
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_images.clear()
        
        # Reload from both sources
        self._load_known_faces()
        self._load_persistent_encodings()
        
        # Remove duplicates (persistent encodings take precedence)
        self._remove_duplicate_encodings()
        
        logger.info(f"Refreshed faces: {len(self.known_face_names)} total")
        
    def get_known_faces_summary(self):
        """Get summary of known faces"""
        return {
            'total_faces': len(self.known_face_names),
            'face_names': self.known_face_names.copy(),
            'encodings_count': len(self.known_face_encodings)
        }
        
    def get_face_encodings(self):
        """Get current face encodings"""
        return self.known_face_encodings.copy()
        
    def get_face_names(self):
        """Get current face names"""
        return self.known_face_names.copy() 
        
    def _remove_duplicate_encodings(self):
        """Remove duplicate encodings, keeping persistent ones"""
        try:
            seen_names = set()
            unique_encodings = []
            unique_names = []
            unique_images = {}
            
            # Process in reverse order to keep persistent encodings (loaded last)
            for i in range(len(self.known_face_names) - 1, -1, -1):
                name = self.known_face_names[i]
                if name not in seen_names:
                    seen_names.add(name)
                    unique_encodings.insert(0, self.known_face_encodings[i])
                    unique_names.insert(0, name)
                    if name in self.known_face_images:
                        unique_images[name] = self.known_face_images[name]
                        
            self.known_face_encodings = unique_encodings
            self.known_face_names = unique_names
            self.known_face_images = unique_images
            
        except Exception as e:
            logger.error(f"Error removing duplicate encodings: {e}") 