import queue
n, b = list(map(int, input().split()))


class Task:
    def __init__(self, time: int, duration: int, index: int) -> None:
        super().__init__()
        self.time = time
        self.duration = duration
        self.index = index


remaining = queue.Queue()
running = False
finish_time = 0


def run_task(remaining: queue.Queue, finish_time: int):
    task_to_run = remaining.get()
    finish_time = max(finish_time, task_to_run.time) + task_to_run.duration
    result[task_to_run.index] = finish_time
    return finish_time, result


result = {}
for i in range(n):
    time, duration = list(map(int, input().split()))
    task = Task(time, duration, index=i)
    result.update({i: 0})
    if task.time > finish_time and remaining.empty():
        running = True
        finish_time = task.time + task.duration
        result[i] = finish_time
    else:
        if task.time >= finish_time and not remaining.empty():
            finish_time, result = run_task(
                remaining=remaining, finish_time=finish_time)
        if remaining.qsize() < b:
            remaining.put(task)
        else:
            result[i] = - 1
while not remaining.empty():
    finish_time, result = run_task(
        remaining=remaining, finish_time=finish_time)
for key in result:
    print(result.get(key), end=' ')
