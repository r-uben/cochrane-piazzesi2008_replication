import sys
from pathlib import Path

def add_src_folder_in_path():
    # Take the path of mains' folder:
    SCRIPT_DIR = Path(__file__).parent
    # Go to project's folder:
    CODE_DIR = SCRIPT_DIR.parent
    # Add it to system's paths:
    sys.path.append(str(CODE_DIR))


# Do it:
add_src_folder_in_path()