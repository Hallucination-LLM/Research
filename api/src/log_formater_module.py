import logging
import colorama
from colorama import Fore  # You'll need to install this package: `pip install colorama`

colorama.init(autoreset=True)

class ColoredFormatter(logging.Formatter):

    COLORS = {
        'DEBUG': Fore.LIGHTBLUE_EX,
        'INFO': Fore.LIGHTGREEN_EX,
        'WARNING': Fore.LIGHTYELLOW_EX,
        'ERROR': Fore.LIGHTRED_EX
    }

    def format(self, record):

        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, colorama.Fore.RESET)
        
        # Color everything apart from the message
        parts = log_message.split(record.message, 1)
        colored_parts = [f"{color}{part}{colorama.Fore.RESET}" if part else "" for part in parts]

        return colored_parts[0] + record.message + colored_parts[1]