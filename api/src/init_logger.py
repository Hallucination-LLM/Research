import logging 
from src.log_formater_module import ColoredFormatter\

def init_logger(
        logger_level: str = "INFO", 
        logger_format: str = None
):

    app_logger = logging.getLogger()

    if not app_logger.handlers:

        app_logger.setLevel(logger_level)

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(ColoredFormatter(logger_format))
        app_logger.addHandler(stream_handler)

    app_logger.info("Logger initialized correctly.")
    return app_logger