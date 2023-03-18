from random import random
from time import sleep
from multiprocessing import current_process
from multiprocessing import Process
from multiprocessing import Queue
from logging.handlers import QueueHandler
import logging
 
# executed in a process that performs logging
def logger_process(queue):
    logger = logging.getLogger('app')
    logger.addHandler(logging.FileHandler('log-parallel-final.txt'))
    logger.setLevel(logging.DEBUG)
    while True:
        message = queue.get()
        if message is None:
            break
        logger.handle(message)
 
# task to be executed in child processes
def task(queue):
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    
    process = current_process()
    logger.info(f'Child {process.name} starting.')
    for i in range(5):
        logger.debug(f'Child {process.name} step {i}.')
        sleep(random())
    logger.info(f'Child {process.name} done.')
 
# protect the entry point
if __name__ == '__main__':
    
    queue = Queue()
    logger = logging.getLogger('app')
    logger.addHandler(QueueHandler(queue))
    logger.setLevel(logging.DEBUG)
    logger_p = Process(target=logger_process, args=(queue,))
    logger_p.start()
    logger.info('Main process started.')

    processes = [Process(target=task, args=(queue,)) for _ in range(5)]
    # start child processes
    for process in processes:
        process.start()
    # wait for child processes to finish
    for process in processes:
        process.join()
    # report final message
    logger.info('Main process done.')
    # shutdown the queue correctly
    queue.put(None)