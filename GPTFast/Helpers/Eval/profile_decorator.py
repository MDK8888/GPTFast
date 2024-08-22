# profiling.py

import functools
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def profile_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check if this is an instance method
        self = args[0] if args and hasattr(args[0], 'profile_enabled') else None
        profile_enabled = getattr(self, 'profile_enabled', False) if self else False

        if not profile_enabled:
            with torch.no_grad():
                return func(*args, **kwargs)

        try:
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function(func.__name__):
                    with torch.no_grad():
                        result = func(*args, **kwargs)

            print(f"Profiling results for {func.__name__}:")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            prof.export_chrome_trace(f"{func.__name__}_trace.json")
        except RuntimeError as e:
            if "Can't disable Kineto profiler when it's not running" in str(e):
                print(f"Warning: Profiler was not running for {func.__name__}. Running without profiling.")
                with torch.no_grad():
                    result = func(*args, **kwargs)
            else:
                raise

        return result

    return wrapper