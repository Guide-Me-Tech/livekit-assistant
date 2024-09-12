from utils.printing import green, red, reset
import time

def timer(func):
    def wrapper(*arg, **kwargs):

        start = time.time()
        result = func(*arg, **kwargs)
        end = time.time()
        print(
            f"           {green} {func.__name__} took to complete:  {red} {end - start}{reset}          "
        )
        return result

    return wrapper