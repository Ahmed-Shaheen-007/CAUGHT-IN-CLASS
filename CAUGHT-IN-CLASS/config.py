import json
import os
import logging
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class ConfigManager(QObject):
    """Manages application configuration and theme switching"""
    
    config_updated = pyqtSignal(dict)  # Emit when config is updated
    
    def __init__(self, config_file="config.json"):
        super().__init__()
        self.config_file = config_file
        self.default_config = {
            'theme': 'light',
            'sound_enabled': True,
            'camera_index': 0,
            'frame_skip': 1,
            'detection_interval': 1.5,  # seconds between face detection processing
            'auto_save_interval': 60,  # seconds
            'thumbnail_size': 80,
            'window_geometry': {
                'width': 1200,
                'height': 800,
                'x': 100,
                'y': 100
            }
        }
        self.config = self.load_config()
        
    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    
                # Merge with default config to ensure all keys exist
                config = self.default_config.copy()
                config.update(loaded_config)
                logger.info(f"Configuration loaded from {self.config_file}")
                return config
            else:
                logger.info("No config file found, using defaults")
                return self.default_config.copy()
                
        except Exception as e:
            logger.error(f"Error loading config: {e}, using defaults")
            return self.default_config.copy()
            
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
            
    def get(self, key, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
        
    def set(self, key, value):
        """Set configuration value and save"""
        self.config[key] = value
        self.save_config()
        self.config_updated.emit(self.config)
        
    def get_theme_stylesheet(self, theme=None):
        """Get stylesheet for specified theme"""
        if theme is None:
            theme = self.config.get('theme', 'light')
            
        if theme == 'dark':
            return self._get_dark_theme()
        else:
            return self._get_light_theme()
            
    def _get_light_theme(self):
        """Get light theme stylesheet"""
        return """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QWidget {
            background-color: white;
            color: #333333;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QFrame {
            background-color: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        QLabel {
            color: #333333;
        }
        QTextEdit {
            border: 1px solid #cccccc;
            border-radius: 4px;
            background-color: white;
            color: #333333;
        }
        QScrollArea {
            border: 1px solid #e0e0e0;
            background-color: white;
        }
        QStatusBar {
            background-color: #f0f0f0;
            color: #333333;
        }
        QToolBar {
            background-color: #f8f8f8;
            border: 1px solid #e0e0e0;
        }
        QMenuBar {
            background-color: #f8f8f8;
            color: #333333;
        }
        QMenuBar::item:selected {
            background-color: #0078d4;
            color: white;
        }
        """
        
    def _get_dark_theme(self):
        """Get dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #2d2d30;
        }
        QWidget {
            background-color: #2d2d30;
            color: #ffffff;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: bold;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #3e3e42;
            color: #666666;
        }
        QFrame {
            background-color: #3e3e42;
            border: 1px solid #555555;
            border-radius: 8px;
        }
        QLabel {
            color: #ffffff;
        }
        QTextEdit {
            border: 1px solid #555555;
            border-radius: 4px;
            background-color: #3e3e42;
            color: #ffffff;
        }
        QScrollArea {
            border: 1px solid #555555;
            background-color: #3e3e42;
        }
        QStatusBar {
            background-color: #3e3e42;
            color: #ffffff;
        }
        QToolBar {
            background-color: #3e3e42;
            border: 1px solid #555555;
        }
        QMenuBar {
            background-color: #3e3e42;
            color: #ffffff;
        }
        QMenuBar::item:selected {
            background-color: #0078d4;
            color: white;
        }
        QMenu {
            background-color: #3e3e42;
            color: #ffffff;
            border: 1px solid #555555;
        }
        QMenu::item:selected {
            background-color: #0078d4;
        }
        """
        
    def toggle_theme(self):
        """Toggle between light and dark themes"""
        current_theme = self.config.get('theme', 'light')
        new_theme = 'dark' if current_theme == 'light' else 'light'
        self.set('theme', new_theme)
        logger.info(f"Theme switched to {new_theme}")
        return new_theme
        
    def toggle_sound(self):
        """Toggle sound on/off"""
        current_sound = self.config.get('sound_enabled', True)
        new_sound = not current_sound
        self.set('sound_enabled', new_sound)
        logger.info(f"Sound {'enabled' if new_sound else 'disabled'}")
        return new_sound
        
    def update_window_geometry(self, x, y, width, height):
        """Update window geometry in config"""
        self.config['window_geometry'] = {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
        self.save_config()
        
    def get_window_geometry(self):
        """Get saved window geometry"""
        return self.config.get('window_geometry', self.default_config['window_geometry'])
        
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self.save_config()
        self.config_updated.emit(self.config)
        logger.info("Configuration reset to defaults")
        
    def get_config_summary(self):
        """Get configuration summary"""
        return {
            'theme': self.config.get('theme', 'light'),
            'sound_enabled': self.config.get('sound_enabled', True),
            'camera_index': self.config.get('camera_index', 0),
            'frame_skip': self.config.get('frame_skip', 1),
            'auto_save_interval': self.config.get('auto_save_interval', 60),
            'thumbnail_size': self.config.get('thumbnail_size', 80)
        } 