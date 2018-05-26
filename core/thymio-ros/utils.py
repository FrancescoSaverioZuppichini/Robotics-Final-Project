def callback(fun,*args, **kwargs):
    def delay(x):
        fun(x, *args, **kwargs)
    return delay


class Params:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z