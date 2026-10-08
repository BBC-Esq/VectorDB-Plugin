import faulthandler
faulthandler.enable(all_threads=True)

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from core.gpu_guard import hide_unsupported_gpus
hide_unsupported_gpus()

from core.native_preload import preload_native_libraries
preload_native_libraries()

from core.cuda_paths import set_cuda_paths
set_cuda_paths()

if __name__ == '__main__':
    from gui.main_window import main
    main()
