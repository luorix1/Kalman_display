import threading
from threading import Thread
import vid
import realtimeCV

lock = threading.Lock()
def one(): vid.vid(lock)
def two(): realtimeCV.realtimeCV(lock)

Thread(target=one).start()
Thread(target=two).start()