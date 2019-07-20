# -*- coding: UTF-8 -*-
from time import time

class Time(object):
    def __init__(self,):
        self.running = False

    @staticmethod
    def now():
        return time()

    @staticmethod
    def nowStr():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

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




