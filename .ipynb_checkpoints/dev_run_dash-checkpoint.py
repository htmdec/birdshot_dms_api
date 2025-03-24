import os
from birdshot import show_plot

# TMP_URL is only needed if using Jupyter proxy mode
os.environ["TMP_URL"] = "localhost:8888"
show_plot()
