import time

class runtimeTimer(object):


    def __init__(self):
        self.starttime = time.time()

    def start(self):
        self.starttime = time.time()


    def stop(self):
        return time.time() - self.starttime