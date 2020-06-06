# 一些通用的工具，如自己写的tqdm
import time


# 自己写的tqdm函数，原来的总是有各种各样的问题
def tqdm(data, desc=''):
    last_time = time.time()
    max_step = len(data)
    start_step = 1
    last_print_time = time.time()
    for i in data:
        new_time = time.time()
        iter_time = new_time - last_time
        last_time = new_time
        speed_msg = '%.2f iter/s' % (1 / iter_time) if iter_time < 1 else '%.2f s/iter' % iter_time
        time_need = iter_time * (max_step - start_step)
        hour_need = time_need / 3600
        time_need_msg = '%.2f' % hour_need
        print_time_now = time.time()
        if print_time_now - last_print_time > 0.3:
            print('\r' + desc + ' :' + str(start_step) + '/' +
                  str(max_step) + ' 还需' + time_need_msg + '小时；' + '速度：' + speed_msg, end='', flush=True)
            last_print_time = print_time_now
        start_step += 1
        yield i

# 自己写的一个trange函数, tqdm总出问题
def trange(start_step, max_step, desc):
    last_time = time.time()
    for i in range(start_step + 1, max_step + 1):
        new_time = time.time()
        time_msg = '%.2f' % ((new_time - last_time) * (max_step - i + 1))
        last_time = new_time
        print('\r' + desc + ' :' + str(i) + '/' + str(max_step) + ' 还需' + time_msg + '秒', flush=True)
        yield i