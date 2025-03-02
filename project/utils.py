import time


def execution_time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Time elapsed: {execution_time}")
        print(f"Start time: {start_time}\nEnd time: {end_time}")

        return result

    return wrapper
