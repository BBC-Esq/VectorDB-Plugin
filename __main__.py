import faulthandler
faulthandler.enable(all_threads=True)

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from core.utilities import set_cuda_paths
set_cuda_paths()

from gui.main_window import main
main()
