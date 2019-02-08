#!/usr/bin/env python3
# coding: utf8

"""
TODO: Write Documentation
"""

import glob
from queue import Queue
import os
import threading
import multiprocessing
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
        

    def convert_tif_to_np(self, data_dir: str, parallelism="THREADING"):
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
        parallelism : str
                      Should be either THREADING or MULTIPROCESSING for selecting 
                      which library to use for parallel computing, multiprocessing
                      has higher initial overhead, but it can get by python's 
                      global interpreter lock, so can have higher utilization.
        Raises
        ------
        FileNotFoundError 
            If no .tif files cannot be located
        ValueError
            If parallelism is not one of THREADING or MULTIPROCESSING
        """
        start = time.time()
        extension = "*.tif"
        if data_dir[-1] != '/':
            extension = "/*.tif"

        files = glob.glob(data_dir + extension)
        if files == []: 
            raise FileNotFoundError(f"Couldn't locate any .tif files from {data_dir}")

        if parallelism == "THREADING":
            self._queue = Queue() 
            print(f"Located {len(files)} .tif files, creating {self.num_threads} threads.")
            self._tqdm = tqdm(total=len(files))
            threads = list()
            # Create Threads
            for thread_id in range(self.num_threads):
                t = threading.Thread(target=self._tif_file_converter)
                t.daemon = True  
                t.name = thread_id
                threads.append(t)
                t.start()

            for f in files:
                self._queue.put(f)
            self._queue.join()

            list(map(lambda t: t.join(), threads))  # Joins all the threads together
            self._tqdm = None

        elif parallelism == "MULTIPROCESSING":
            print(f"Located {len(files)} .tif files, creating {self.num_threads} processes.")
            self._queue = multiprocessing.Queue()
            processes = list()
            # Create processes
            for pid in range(self.num_threads):
                p = multiprocessing.Process(target=self._tif_file_converter)    
                p.daemon = True
                processes.append(p)
                p.start()

            for f in files:
                self._queue.put(f)
            list(map(lambda p: p.join(), processes))
        else:
            raise ValueError(f"Invalid value for parallelism: {parallelism}")
            
        print("Time Elapsed: {:.3f} seconds\n".format(time.time() - start))
         
    
    def _tif_file_converter(self):
        """
        TODO: Write documentation for _tif_file_converter
        """
        while True:
            tif_file = self._queue.get()
            img = Image.open(tif_file)

            np_image = np.array(img, dtype = np.float32)
            np.save(tif_file.replace('.tif','.npy'), np_image)   

            if isinstance(self._queue, Queue):
                self._queue.task_done()
                with self._lock:
                    self._tqdm.update(1)
            else:
                print(f"Completed processing: {tif_file}")

            if self._queue.empty():
                break


if __name__ == "__main__":
    p = Preprocesser(num_threads=5)
    p.convert_tif_to_np("tif_images", parallelism="MULTIPROCESSING")
