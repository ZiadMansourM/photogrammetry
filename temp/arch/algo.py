import time

def log_to_file(file_name: str, message: str):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    with open(file_name, "a") as f:
        f.write(f"{log_message}\n")

def timeit(func):
    # from functools import wraps
    # @wraps(func)
    def wrapper(*args, **kwargs):
        # img_set_name = kwargs.get('img_set_name', 'default')
        img_set_name = kwargs['img_set_name']
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Started {func.__name__}...")
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        log_to_file(f"data/{img_set_name}/logs/tune.log", f"Done {func.__name__} took {end_time - start_time:,} seconds to execute.")
        return result
    return wrapper

@timeit
def partial_sleep(*args, **kwargs):
    print("from inside _partial_sleep")
    time.sleep(3)
    print("Partial sleep")

@timeit
def algo_logic(*args, **kwargs):
    print("from inside _algo_logic")
    time.sleep(2)
    partial_sleep(**kwargs)
    print("Algo logic")

def run(img_set_name: str):
    print("from inside run")
    algo_logic(img_set_name=img_set_name)