# -*- coding: UTF-8 -*-
from time import time,localtime,strftime

class Time(object):
    def __init__(self,):
        self.running = False

    @staticmethod
    def now():
        return time()

    @staticmethod
    def now_str():
        return strftime("%Y-%m-%d %H:%M:%S", localtime(time()))

    @staticmethod
    def is_before(t):
        return time() - t < 0

    @staticmethod
    def is_after(t):
        return time() - t > 0

    def start(self):
        self.running = True
        self._t_start = time()
        self._t_last = time()

    def since_start(self):
        if not self.running:
            return 0
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last(self):
        if not self.running:
            return 0
        self._t_last = time()
        return self._t_last - self._t_start




