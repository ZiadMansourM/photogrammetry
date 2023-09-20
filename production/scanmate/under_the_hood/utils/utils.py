import pickle
import time
from typing import Final

from data_structures import Images

HOST_PATH: Final[str] = "../../.."

def log_to_file(file_name: str, message: str, **kwargs):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    image_set_name = kwargs['image_set_name']
    base_path: str = f"{HOST_PATH}/data/{image_set_name}"
    with open(f"{base_path}/{file_name}", "a") as f:
        f.write(f"{log_message}\n")


def print_size(file_name: str, obj, obj_name="N/A"):
    from pympler import asizeof
    memory_usage = asizeof.asizeof(obj)
    # Convert memory usage to a more readable format
    if memory_usage < 1024:
        memory_usage_str = f"{memory_usage} bytes"
    elif memory_usage < 1024 ** 2:
        memory_usage_str = f"{memory_usage / 1024} KB"
    elif memory_usage < 1024 ** 3:
        memory_usage_str = f"{memory_usage / (1024 ** 2)} MB"
    else:
        memory_usage_str = f"{memory_usage / (1024 ** 3)} GB"
    # Print the memory usage and object name
    log_to_file(file_name, f"Memory usage of {obj_name}: {memory_usage_str}")


def timeit(func):
    def wrapper(*args, **kwargs):
        log_to_file("logs/tune.log", f"Started {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_to_file("logs/tune.log", f"Done {func.__name__} took {end_time - start_time:,} seconds to execute.")
        return result
    return wrapper

def dump_images_bak(images_file_path: str, images: Images, **kwargs) -> None:
    """ Dump images to a file """
    image_set_name = kwargs['image_set_name']
    base_path: str = f"{HOST_PATH}/data/{image_set_name}"
    with open(f"{base_path}/{images_file_path}", "wb") as file:
        pickle.dump(images, file)


def load_images_bak(images_file_path: str, **kwargs) -> Images:
    """ Load images from a file """
    image_set_name = kwargs['image_set_name']
    base_path: str = f"{HOST_PATH}/data/{image_set_name}"
    with open(f"{base_path}/{images_file_path}", "rb") as file:
        images = pickle.load(file)
    return images