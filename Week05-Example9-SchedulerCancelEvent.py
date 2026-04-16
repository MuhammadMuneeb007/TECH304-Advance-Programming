import sched
import time


scheduler = sched.scheduler(time.time, time.sleep)


def task1():
    print("Task 1 executed at:", time.time())


def task2():
    print("Task 2 executed at:", time.time())


event1 = scheduler.enter(delay=1, priority=1, action=task1)
event2 = scheduler.enter(delay=1, priority=2, action=task2)

print("Events scheduled. Current time:", time.time())

try:
    scheduler.cancel(event1)
    print("Task 1 canceled.")
except ValueError:
    print("Task 1 was not found in the event queue.")

print("Running the scheduler...")
scheduler.run()
