"""Keep routine tests independent of developer-local dotenv configuration."""

import os


os.environ["PYTHON_DOTENV_DISABLED"] = "1"
