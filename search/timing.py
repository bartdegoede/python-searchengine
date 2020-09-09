import time

def timing(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        execution_time = end - start
        if execution_time < 0.001:
            print(f'{method.__name__} took {execution_time*1000} milliseconds')
        else:
            print(f'{method.__name__} took {execution_time} seconds')

        return result
    return timed
