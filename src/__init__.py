import logging
import os
from datetime import datetime

# Create logs directory
log_dir = "artifacts/logs"
os.makedirs(log_dir, exist_ok=True)

# Log file path with timestamp
log_file = os.path.join(log_dir, f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s: %(message)s]",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),  # Use UTF-8 encoding
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NoiseReductionAutoencoder")