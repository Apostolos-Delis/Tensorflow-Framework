#!/usr/bin/env python3
# coding: utf8

"""
TODO: Write Documentation
"""

import glob
from queue import Queue
import os
import threading
import time

import numpy as np
from PIL import Image
from tqdm import tqdm, trange


class Preprocesser(object):
    """
    Preprocessor Object for processing the data before it is used 
    in training.

    Atributes
    --------- 
    num_threads : int 
                  Stores the number of threads that will be used.
    """

    def __init__(self, num_threads=8):
        self._queue = None
        self._tqdm = None
        self.num_threads = num_threads
        self._lock = threading.Lock()
        

    def convert_tif_to_np(self, data_dir: str):
        """
        Takes a data_dir, locates all the valid *.tif files to search and then 
        using multithreading, converts them.

        Parameters
        ----------
        data_dir : str 
                      The string containing the directory name, this can include 
                      wildcards for multiple directories, for example: images/*/ 
                      which will select all files with a .tif extension in any
                      of the directories under images
        Raises
        ------
        FileNotFoundError 
            If no .tif files cannot be located
        """
        start = time.time()
        extension = "*.tif"
        if data_dir[-1] != '/':
            extension = "/*.tif"

        files = glob.glob(data_dir + extension)
        if files == []: 
            raise FileNotFoundError(f"Couldn't locate any .tif files from {data_dir}")
        self._queue = Queue() 

        print(f"Located {len(files)} .tif files, creating {self.num_threads} threads.")
        self._tqdm = tqdm(total=len(files))
        threads = list()
        for thread_id in range(self.num_threads):
            t = threading.Thread(target=self._tif_file_converter)
            threads.append(t)
            t.daemon = True  
            t.name = thread_id
            t.start()

        for f in files:
            self._queue.put(f)

        self._queue.join()
        list(map(lambda t: t.join(), threads))
        self._tqdm = None
        print("Time Elapsed: {:.3f} seconds\n".format(time.time() - start))
         
    
    def _tif_file_converter(self):
        while True:
            tif_file = self._queue.get()
            # print(tif_file, self._queue.qsize())

            img = Image.open(tif_file)
            np_image = np.array(img, dtype = np.float32)
            np.save(tif_file.replace('.tif','.npy'), np_image)   

            with self._lock:
                self._tqdm.update(1)
            self._queue.task_done()
            if self._queue.empty():
                break


if __name__ == "__main__":
    p = Preprocesser(num_threads=15)
    p.convert_tif_to_np("tif_images")
