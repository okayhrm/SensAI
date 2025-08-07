import logging

# Get a logger instance. It's good practice to name loggers,
# but for a simple utility, __name__ is fine.
logger = logging.getLogger(__name__)

# Set the logging level for this logger.
# logging.DEBUG will show ALL messages (DEBUG, INFO, WARNING, ERROR, CRITICAL).
# Use logging.INFO if you only want informational messages and above.
logger.setLevel(logging.DEBUG)

# Create a console handler (StreamHandler)
# This sends log records to the console (sys.stdout or sys.stderr).
console_handler = logging.StreamHandler()

# Set the level for the handler. This controls what the handler actually outputs.
# Setting it to DEBUG means the console will show all messages passed to it.
console_handler.setLevel(logging.DEBUG)

# Create a formatter for the log messages.
# This defines the layout of your log messages.
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the formatter to the handler.
console_handler.setFormatter(formatter)

# Add the handler to the logger.
# The 'if not logger.handlers:' check prevents adding duplicate handlers
# if the module is reloaded (e.g., by Uvicorn's --reload).
if not logger.handlers:
    logger.addHandler(console_handler)

# Optional: Disable propagation to the root logger.
# If you don't do this, messages might be duplicated if the root logger also has handlers.
logger.propagate = False