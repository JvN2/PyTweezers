import multiprocessing as mp
import numpy as np
import collections

from  modules.TweezersSupport import time_it


Msg = collections.namedtuple('Msg', ['event', 'args'])

class BaseProcess(mp.Process):
    """A process backed by an internal queue for simple one-way message passing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = mp.Queue()

    def send(self, event, *args):
        """Puts the event and args as a `Msg` on the queue
        """
        msg = Msg(event, args)
        self.queue.put(msg)

    def dispatch(self, msg):
        event, args = msg

        handler = getattr(self, "do_%s" % event, None)
        if not handler:
            raise NotImplementedError("Process has no handler for [%s]" % event)

        handler(*args)

    def run(self):
        while True:
            msg = self.queue.get()
            self.dispatch(msg)


class MyProcess(BaseProcess):
    def do_helloworld(self, arg1, arg2):
        print(arg1, arg2)

    @time_it
    def do_howmany(self, row, minimum, maximum):
        """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
        count = 0
        for n in row:
            count += np.sum(np.asarray(n) < maximum )
            # if minimum <= n <= maximum:
            #     count = count + 1
        print(count)
        return count

if __name__ == "__main__":
    process = MyProcess()
    process.start()
    process.send('helloworld', 'hello', 'world')

    data = np.random.randint(0, 10, size=[int(1e6), 5]).tolist()

    process.send('howmany', data, 2, 4)
    process.send('howmany', data, 2, 4)