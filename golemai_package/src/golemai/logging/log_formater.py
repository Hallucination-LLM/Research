import logging

# You'll need to install this package: `pip install colorama`
import colorama
from colorama import Fore
from golemai.config import LOGGER_FORMAT

colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):

    COLORS = {
        "DEBUG": Fore.LIGHTBLUE_EX,
        "INFO": Fore.LIGHTGREEN_EX,
        "WARNING": Fore.LIGHTYELLOW_EX,
        "ERROR": Fore.LIGHTRED_EX,
    }

    def format(self, record):

        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, colorama.Fore.RESET)

        # Color everything apart from the message
        parts = log_message.split(record.message, 1)
        colored_parts = [
            f"{color}{part}{colorama.Fore.RESET}" if part else ""
            for part in parts
        ]

        return colored_parts[0] + record.message + colored_parts[1]


def init_logger(logger_level: str = "INFO"):
    """Initialize the logger with the provided level."""

    app_logger = logging.getLogger()

    if not app_logger.handlers:

        app_logger.setLevel(logger_level)

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(ColoredFormatter(LOGGER_FORMAT))
        app_logger.addHandler(stream_handler)

    return app_logger
