import pandas as pd
import os
import logging
from datetime import datetime
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class AttendanceLogger(QObject):
    """Handles attendance logging to Excel files"""
    
    log_updated = pyqtSignal(str)  # Emit when log is updated
    log_error = pyqtSignal(str)    # Emit when logging fails
    
    def __init__(self, master_file="attendance_log.xlsx"):
        super().__init__()
        self.master_file = master_file
        self.columns = ['Name', 'Date', 'Time', 'Status']
        
    def append_to_master_log(self, name, status, timestamp=None):
        """Append a new attendance record to the master log"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
                
            # Create new record
            new_record = {
                'Name': name,
                'Date': timestamp.strftime('%Y-%m-%d'),
                'Time': timestamp.strftime('%H:%M:%S'),
                'Status': status
            }
            
            # Check if master file exists
            if os.path.exists(self.master_file):
                try:
                    # Read existing data
                    df_existing = pd.read_excel(self.master_file)
                    
                    # Verify columns match
                    if list(df_existing.columns) != self.columns:
                        logger.warning(f"Column mismatch in {self.master_file}, recreating file")
                        df_existing = pd.DataFrame(columns=self.columns)
                        
                except Exception as e:
                    logger.warning(f"Error reading existing master file: {e}, recreating file")
                    df_existing = pd.DataFrame(columns=self.columns)
            else:
                # Create new file with headers
                df_existing = pd.DataFrame(columns=self.columns)
                
            # Append new record
            df_new = pd.DataFrame([new_record])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Write to master file
            with pd.ExcelWriter(self.master_file, engine='openpyxl') as writer:
                df_combined.to_excel(writer, sheet_name='Attendance', index=False)
                
            logger.info(f"Successfully logged: {name} - {status}")
            self.log_updated.emit(f"Logged: {name} ({status})")
            
            return True
            
        except Exception as e:
            error_msg = f"Error logging attendance: {str(e)}"
            logger.error(error_msg)
            self.log_error.emit(error_msg)
            return False
            
    def get_master_log_summary(self):
        """Get summary of master log"""
        try:
            if not os.path.exists(self.master_file):
                return {
                    'total_records': 0,
                    'last_entry': None,
                    'file_exists': False
                }
                
            df = pd.read_excel(self.master_file)
            
            if df.empty:
                return {
                    'total_records': 0,
                    'last_entry': None,
                    'file_exists': True
                }
                
            # Get last entry
            last_entry = df.iloc[-1].to_dict() if len(df) > 0 else None
            
            return {
                'total_records': len(df),
                'last_entry': last_entry,
                'file_exists': True
            }
            
        except Exception as e:
            logger.error(f"Error reading master log summary: {e}")
            return {
                'total_records': 0,
                'last_entry': None,
                'file_exists': False,
                'error': str(e)
            }
            
    def export_session_summary(self, session_records, filename="session_summary.xlsx"):
        """Export current session records to separate file"""
        try:
            if not session_records:
                logger.warning("No session records to export")
                return False
                
            # Convert session records to DataFrame
            df_session = pd.DataFrame(session_records)
            
            # Ensure columns match
            if list(df_session.columns) != self.columns:
                logger.warning("Session records columns don't match expected format")
                return False
                
            # Write to file
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_session.to_excel(writer, sheet_name='Session', index=False)
                
            logger.info(f"Session summary exported to {filename}")
            return True
            
        except Exception as e:
            error_msg = f"Error exporting session summary: {str(e)}"
            logger.error(error_msg)
            self.log_error.emit(error_msg)
            return False
            
    def get_attendance_stats(self, start_date=None, end_date=None):
        """Get attendance statistics for a date range"""
        try:
            if not os.path.exists(self.master_file):
                return {}
                
            df = pd.read_excel(self.master_file)
            
            if df.empty:
                return {}
                
            # Convert Date column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter by date range if specified
            if start_date:
                df = df[df['Date'] >= start_date]
            if end_date:
                df = df[df['Date'] <= end_date]
                
            # Calculate statistics
            stats = {
                'total_records': len(df),
                'known_faces': len(df[df['Status'] == 'Present']),
                'unknown_faces': len(df[df['Status'] == 'Unknown']),
                'unique_people': df[df['Status'] == 'Present']['Name'].nunique(),
                'date_range': {
                    'start': df['Date'].min().strftime('%Y-%m-%d') if not df.empty else None,
                    'end': df['Date'].max().strftime('%Y-%m-%d') if not df.empty else None
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating attendance stats: {e}")
            return {}
            
    def backup_master_log(self, backup_dir="backups"):
        """Create a backup of the master log"""
        try:
            if not os.path.exists(self.master_file):
                logger.warning("No master log file to backup")
                return False
                
            # Create backup directory
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"attendance_log_backup_{timestamp}.xlsx"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file
            import shutil
            shutil.copy2(self.master_file, backup_path)
            
            logger.info(f"Master log backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up master log: {e}")
            return False 