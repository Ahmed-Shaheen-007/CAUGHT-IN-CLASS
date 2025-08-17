import os
import logging
from PyQt5.QtCore import QObject, pyqtSignal, QUrl
from PyQt5.QtMultimedia import QSoundEffect, QMediaPlayer, QMediaContent

logger = logging.getLogger(__name__)

class AudioManager(QObject):
    """Manages audio notifications for the application"""
    
    audio_ready = pyqtSignal(bool)  # Emit when audio is ready
    audio_error = pyqtSignal(str)   # Emit audio errors
    
    def __init__(self, sounds_dir="assets/sounds"):
        super().__init__()
        self.sounds_dir = sounds_dir
        self.sound_enabled = True
        self.sound_effects = {}
        
        # Create sounds directory if it doesn't exist
        os.makedirs(sounds_dir, exist_ok=True)
        
        # Initialize sound effects
        self._init_sound_effects()
        
    def _init_sound_effects(self):
        """Initialize sound effects"""
        try:
            # Define sound files
            sound_files = {
                'known_face': 'notify_known.wav',
                'unknown_face': 'notify_unknown.wav',
                'error': 'error.wav',
                'success': 'success.wav'
            }
            
            # Create default sound files if they don't exist
            self._create_default_sounds()
            
            # Load sound effects
            for sound_name, filename in sound_files.items():
                filepath = os.path.join(self.sounds_dir, filename)
                if os.path.exists(filepath):
                    sound_effect = QSoundEffect()
                    sound_effect.setSource(QUrl.fromLocalFile(filepath))
                    sound_effect.setVolume(0.7)
                    self.sound_effects[sound_name] = sound_effect
                    logger.info(f"Loaded sound effect: {sound_name}")
                else:
                    logger.warning(f"Sound file not found: {filepath}")
                    
        except Exception as e:
            logger.error(f"Error initializing sound effects: {e}")
            self.audio_error.emit(f"Audio initialization error: {str(e)}")
            
    def _create_default_sounds(self):
        """Create default sound files if they don't exist"""
        try:
            # This is a placeholder - in a real application, you would have actual sound files
            # For now, we'll just ensure the directory exists
            default_sounds = [
                'notify_known.wav',
                'notify_unknown.wav',
                'error.wav',
                'success.wav'
            ]
            
            for sound_file in default_sounds:
                filepath = os.path.join(self.sounds_dir, sound_file)
                if not os.path.exists(filepath):
                    logger.info(f"Creating placeholder for sound file: {sound_file}")
                    # Create an empty file as placeholder
                    with open(filepath, 'w') as f:
                        f.write("# Placeholder sound file")
                        
        except Exception as e:
            logger.error(f"Error creating default sounds: {e}")
            
    def set_sound_enabled(self, enabled):
        """Enable or disable sound"""
        self.sound_enabled = enabled
        logger.info(f"Sound {'enabled' if enabled else 'disabled'}")
        
    def play_sound(self, sound_name):
        """Play a specific sound effect"""
        if not self.sound_enabled:
            return
            
        try:
            if sound_name in self.sound_effects:
                sound_effect = self.sound_effects[sound_name]
                if sound_effect.isLoaded():
                    sound_effect.play()
                    logger.debug(f"Playing sound: {sound_name}")
                else:
                    logger.warning(f"Sound effect not loaded: {sound_name}")
            else:
                logger.warning(f"Sound effect not found: {sound_name}")
                
        except Exception as e:
            logger.error(f"Error playing sound {sound_name}: {e}")
            self.audio_error.emit(f"Audio playback error: {str(e)}")
            
    def play_known_face_sound(self):
        """Play sound for known face detection"""
        self.play_sound('known_face')
        
    def play_unknown_face_sound(self):
        """Play sound for unknown face detection"""
        self.play_sound('unknown_face')
        
    def play_error_sound(self):
        """Play sound for errors"""
        self.play_sound('error')
        
    def play_success_sound(self):
        """Play sound for successful operations"""
        self.play_sound('success')
        
    def set_volume(self, volume):
        """Set volume for all sound effects (0.0 to 1.0)"""
        try:
            volume = max(0.0, min(1.0, volume))  # Clamp between 0 and 1
            for sound_effect in self.sound_effects.values():
                sound_effect.setVolume(volume)
            logger.info(f"Volume set to {volume}")
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            
    def get_volume(self):
        """Get current volume level"""
        try:
            if self.sound_effects:
                # Return volume of first sound effect as reference
                first_sound = next(iter(self.sound_effects.values()))
                return first_sound.volume()
            return 0.7  # Default volume
        except Exception as e:
            logger.error(f"Error getting volume: {e}")
            return 0.7
            
    def reload_sounds(self):
        """Reload all sound effects"""
        try:
            # Clear existing sound effects
            self.sound_effects.clear()
            
            # Reinitialize
            self._init_sound_effects()
            
            logger.info("Sound effects reloaded")
            self.audio_ready.emit(True)
            
        except Exception as e:
            logger.error(f"Error reloading sounds: {e}")
            self.audio_error.emit(f"Sound reload error: {str(e)}")
            
    def get_available_sounds(self):
        """Get list of available sound effects"""
        return list(self.sound_effects.keys())
        
    def is_sound_loaded(self, sound_name):
        """Check if a sound effect is loaded"""
        if sound_name in self.sound_effects:
            return self.sound_effects[sound_name].isLoaded()
        return False
        
    def cleanup(self):
        """Clean up audio resources"""
        try:
            for sound_effect in self.sound_effects.values():
                sound_effect.stop()
            self.sound_effects.clear()
            logger.info("Audio manager cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up audio manager: {e}") 