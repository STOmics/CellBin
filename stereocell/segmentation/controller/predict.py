from queue import Queue
from threading import Thread, Event
from controller.processing import f_post_process
import numpy as np
from rich.progress import track


class cl_predict(object):
    def __init__(self, model):
        self._model = model
        self._t_queue_maxsize = 100
        self._t_workdone = Event()
        self._t_queue = Queue(maxsize=self._t_queue_maxsize)

    def _f_productor(self, img_lst):
        self._t_workdone.set()
        for img in track(img_lst, description='Inference'):
            val_sum = np.sum(img)
            if val_sum <= 0.0:
                pred = np.zeros(img.shape, np.uint8)
            else: pred = self._model.f_predict(img)
            self._t_queue.put([pred, val_sum], block=True)
        self._t_workdone.clear()
        return

    def _f_consumer(self, pred_lst):
        while (self._t_workdone.is_set()) or (not self._t_queue.empty()):
            pred, val_sum = self._t_queue.get(block=True)
            if val_sum > 0:
                pred = f_post_process(pred)
            pred_lst.append(pred)
        return

    def _f_clear(self):
        self._t_queue = Queue(maxsize=self._t_queue_maxsize)

    def _run_batch(self, img_lst):
        self._f_clear()
        pred_lst = []
        t_productor = Thread(target=self._f_productor, args=(img_lst,))
        t_consumer = Thread(target=self._f_consumer, args=(pred_lst,))
        t_productor.start()
        t_consumer.start()
        t_productor.join()
        t_consumer.join()
        self._f_clear()
        return pred_lst

    def f_predict(self, img_lst):
        img = img_lst

        if isinstance(img_lst, list):
            if len(img_lst) > 1:
                return self._run_batch(img_lst)
            elif 0 < len(img_lst) < 2:
                img = img_lst[0]

        if np.sum(img) < 1:
            pred = np.zeros(img.shape, np.uint8)
        else:
            pred = self._model.f_predict(img)[0]
        pred = f_post_process(pred)
        return pred
