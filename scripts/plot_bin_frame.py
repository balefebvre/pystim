import os
import tempfile

# from pystim.io.bin import open_file as open_bin_file


name = "fi"
directory = os.path.join(tempfile.gettempdir(), "pystim", name)
filename = "{}.bin"
path = os.path.join(directory, filename)

raise NotImplementedError  # TODO complete.
