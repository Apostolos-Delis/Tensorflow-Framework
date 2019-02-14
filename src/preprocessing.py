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

import scipy.io as sio
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
        

    def convert_to_np(self, data_dir: str, image_type: str, parallelism="THREADING"):
        """
        Takes a data_dir, locates all the valid *.image_type files to search and then 
        using multithreading, converts them.

        Parameters
        ----------
        data_dir :    str 
                      The string containing the directory name, this can include 
                      wildcards for multiple directories, for example: images/*/ 
                      which will select all files with a .tif extension in any
                      of the directories under images
        image_type :  str
                      String that is one of the following: {"tif", "png", "mat"},
                      represents what the original datatype files are called.
        parallelism : str
                      Should be either THREADING or MULTIPROCESSING for selecting 
                      which library to use for parallel computing, multiprocessing
                      has higher initial overhead, but it can get by python's 
                      global interpreter lock, so can have higher utilization.
        Raises
        ------
        FileNotFoundError 
            If no files of image_type could be located
        ValueError
            If parallelism is not one of THREADING or MULTIPROCESSING
            If image_type is not one of the following: mat, tif, or png
        """
        start = time.time()
        if image_type not in {"tif", "png", "mat"}:
            raise ValueError(f"{image_type} is not a valid type!")

        if data_dir[-1] != '/':
            extension = "/*." + image_type
        else:
            extension = "*." + image_type

        files = glob.glob(data_dir + extension)
        # print(files)
        # exit(0)
        if files == []: 
            raise FileNotFoundError(f"Couldn't locate any .{image_type} files from {data_dir}")

        if parallelism == "THREADING":
            self._queue = Queue() 
            print(f"Located {len(files)} .{image_type} files, creating {self.num_threads} threads.")
            self._tqdm = tqdm(total=len(files))
            threads = list()
            # Create Threads
            for thread_id in range(self.num_threads):
                t = threading.Thread(target=self._file_converter, args=(image_type,))
                t.daemon = True  
                t.name = thread_id
                threads.append(t)
                t.start()

            for f in files:
                self._queue.put(f)
            self._queue.join()

            # Joins all the threads together
            for t in threads:
                t.join()
            # list(map(lambda t: t.join(), threads))  
            self._tqdm = None

        elif parallelism == "MULTIPROCESSING":
            print(f"Located {len(files)} .{image_type} files, creating {self.num_threads} processes.")
            self._queue = multiprocessing.Queue()
            processes = list()
            # Create processes
            for pid in range(self.num_threads):
                p = multiprocessing.Process(target=self._file_converter, args=(image_type,))    
                p.daemon = True
                processes.append(p)
                p.start()

            for f in files:
                self._queue.put(f)

            for p in processes:
                p.join()
            # list(map(lambda p: p.join(), processes))
        else:
            raise ValueError(f"Invalid value for parallelism: {parallelism}")
            
        print("Time Elapsed: {:.3f} seconds\n".format(time.time() - start))
         
    
    def _file_converter(self, image_type):
        """
        TODO: Write documentation for _file_converter
        """
        while True:
            file_name = self._queue.get()
            if image_type == "mat":
                img = sio.loadmat(file_name)
                data_key = None
                for key, val in img.items():
                    if isinstance(val, np.ndarray):
                        data_key = key
                np_image = np.array(img[data_key], dtype = np.float32)
            else:
                img = Image.open(file_name)
                np_image = np.array(img, dtype = np.float32)

            np.save(file_name.replace('.'+image_type,'.npy'), np_image)   

            if isinstance(self._queue, Queue):
                self._queue.task_done()
                with self._lock:
                    self._tqdm.update(1)
            else:
                print(f"Completed processing: {file_name}")

            if self._queue.empty():
                break


if __name__ == "__main__":

    p = Preprocesser(num_threads=5)
    p.convert_to_np("mat_files", image_type="mat", parallelism="THREADING")


