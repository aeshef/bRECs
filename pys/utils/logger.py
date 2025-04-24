# utils/logger.py
import logging
import os

class BaseLogger:
    """Base class providing standardized logging setup for all application components"""
    
    def __init__(self, logger_name=None):
        """Initialize with an optional logger name or use class name"""
        self.logger = self.setup_logger(logger_name or self.__class__.__name__)
    
    def setup_logger(self, logger_name):
        """Set up and configure logger with consistent formatting"""
        logger = logging.getLogger(logger_name)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            logs_dir = os.path.join(project_root, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(logs_dir, f'{logger_name}.log'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.setLevel(logging.INFO)
        
        return logger
        
    def log_exception(self, message, exception):
        """Log exception with traceback"""
        self.logger.exception(f"{message}: {str(exception)}")

    def log_start_process(self, process_name):
        """Log standard message for process start"""
        self.logger.info(f"Starting process: {process_name}")

    def log_end_process(self, process_name, success=True):
        """Log standard message for process end"""
        if success:
            self.logger.info(f"Successfully completed: {process_name}")
        else:
            self.logger.warning(f"Process completed with issues: {process_name}")
