import threading 
import time

def thread_task(thread_id, delay): 
	for i in range(5): 
		time.sleep(delay) 
		print(f"Thread {thread_id}: Executing...") 

thread1 = threading.Thread(target=thread_task, args=(1, 1)) 
thread2 = threading.Thread(target=thread_task, args=(2, 20))
 
thread1.start() 
thread2.start() 

thread1.join() 
thread2.join()
print("Both threads have completed execution.")