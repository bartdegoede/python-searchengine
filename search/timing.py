import time

def timing(method):
    """
    Quick and dirty decorator to time functions: it will record the time when
    it's calling a function, record the time when it returns and compute the
    difference. There'll be some overhead, so it's not very precise, but'll
    suffice to illustrate the examples in the accompanying blog post.

    @timing
    def snore():
        print('zzzzz')
        time.sleep(5)

    snore()
    zzzzz
    snore took 5.0011749267578125 seconds
    """
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
