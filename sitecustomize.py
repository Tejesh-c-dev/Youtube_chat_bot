import os
import sys

# Disable writing .pyc files globally for this workspace Python startup.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
