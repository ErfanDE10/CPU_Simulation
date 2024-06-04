import cv2
import numpy as np
import heapq
from collections import deque

class Process:
    def __init__(self, pid, arrival_time, burst_time):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.start_time = None
        self.finish_time = None
        self.waiting_time = 0

def fcfs(processes):
    time = 0
    schedule = []
    for process in processes:
        if time < process.arrival_time:
            time = process.arrival_time
        process.start_time = time
        process.finish_time = time + process.burst_time
        process.waiting_time = time - process.arrival_time
        schedule.append((time, process.pid))
        time += process.burst_time
    return schedule

def sjf(processes):
    time = 0
    schedule = []
    processes = sorted(processes, key=lambda x: (x.arrival_time, x.burst_time))
    ready_queue = []
    i = 0
    while i < len(processes) or ready_queue:
        while i < len(processes) and processes[i].arrival_time <= time:
            heapq.heappush(ready_queue, (processes[i].burst_time, processes[i]))
            i += 1
        if ready_queue:
            burst_time, process = heapq.heappop(ready_queue)
            process.start_time = time
            process.finish_time = time + burst_time
            process.waiting_time = time - process.arrival_time
            schedule.append((time, process.pid))
            time += burst_time
        else:
            time = processes[i].arrival_time
    return schedule

def srt(processes):
    time = 0
    schedule = []
    processes = sorted(processes, key=lambda x: x.arrival_time)
    ready_queue = []
    i = 0
    while i < len(processes) or ready_queue:
        while i < len(processes) and processes[i].arrival_time <= time:
            heapq.heappush(ready_queue, (processes[i].remaining_time, processes[i]))
            i += 1
        if ready_queue:
            remaining_time, process = heapq.heappop(ready_queue)
            schedule.append((time, process.pid))
            time += 1
            process.remaining_time -= 1
            if process.remaining_time > 0:
                heapq.heappush(ready_queue, (process.remaining_time, process))
            else:
                process.finish_time = time
        else:
            time = processes[i].arrival_time
    return schedule

def hrrn(processes):
    time = 0
    schedule = []
    processes = sorted(processes, key=lambda x: x.arrival_time)
    ready_queue = []
    i = 0
    while i < len(processes) or ready_queue:
        while i < len(processes) and processes[i].arrival_time <= time:
            ready_queue.append(processes[i])
            i += 1
        if ready_queue:
            process = max(ready_queue, key=lambda x: (time - x.arrival_time + x.burst_time) / x.burst_time)
            ready_queue.remove(process)
            process.start_time = time
            process.finish_time = time + process.burst_time
            process.waiting_time = time - process.arrival_time
            schedule.append((time, process.pid))
            time += process.burst_time
        else:
            time = processes[i].arrival_time
    return schedule

def rr(processes, quantum):
    time = 0
    schedule = []
    ready_queue = deque(processes)
    while ready_queue:
        process = ready_queue.popleft()
        if process.remaining_time > quantum:
            schedule.append((time, process.pid))
            time += quantum
            process.remaining_time -= quantum
            for p in processes:
                if p.arrival_time <= time and p not in ready_queue and p.remaining_time > 0:
                    ready_queue.append(p)
            ready_queue.append(process)
        else:
            schedule.append((time, process.pid))
            time += process.remaining_time
            process.finish_time = time
            process.remaining_time = 0
    return schedule

def display_text(image, text, position, font_scale=1, color=(255, 255, 255), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def draw_gui(image):
    global current_algorithm, schedule, best_algorithm, process_output, RESULT_X_START
    
    image.fill(0)  
    
    window_width = image.shape[1]
    window_height = image.shape[0]
    
    
    left_section_width = 300
    RESULT_X_START = left_section_width  # Define RESULT_X_START based on left_section_width

    
    for i, alg in enumerate(ALGORITHMS):
        y = i * BUTTON_HEIGHT
        cv2.rectangle(image, (0, y), (left_section_width, y + BUTTON_HEIGHT), (50, 50, 50), -1)
        display_text(image, alg, (10, y + 35), font_scale=0.8)

    
    right_section_x_start = left_section_width + 20
    right_section_y_start = 10
    right_section_width = window_width - right_section_x_start - 20
    right_section_height = window_height - 20

    cv2.rectangle(image, (right_section_x_start, right_section_y_start),
                  (right_section_x_start + right_section_width, right_section_y_start + right_section_height),
                  (70, 70, 70), -1)
    
    
    if current_algorithm:
        display_text(image, f"{current_algorithm} Algorithm:", (right_section_x_start + 10, right_section_y_start + 30), font_scale=1.2)
        for j, (time, pid) in enumerate(schedule):
            display_text(image, f"Time {time}: Process {pid}", (right_section_x_start + 10, right_section_y_start + 60 + j * 30), font_scale=0.8)
    
    
    output_section_y_start = len(ALGORITHMS) * BUTTON_HEIGHT + 20
    output_section_height = 200  

    cv2.rectangle(image, (10, output_section_y_start), (left_section_width - 10, output_section_y_start + output_section_height), (70, 70, 70), -1)

    if best_algorithm:
        display_text(image, f"Best Algorithm: {best_algorithm}", (20, output_section_y_start + 30), font_scale=1, color=(0, 255, 0))
    
    if process_output:
        display_text(image, "Processing Output:", (20, output_section_y_start + 60), font_scale=1)
        for j, line in enumerate(process_output):
            display_text(image, line, (20, output_section_y_start + 90 + j * 30), font_scale=0.8)

def mouse_callback(event, x, y, flags, param):
    global current_algorithm, schedule, best_algorithm, process_output
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < RESULT_X_START:
            index = y // BUTTON_HEIGHT
            if 0 <= index < len(ALGORITHMS):
                current_algorithm = ALGORITHMS[index]
                processes_copy = [Process(p.pid, p.arrival_time, p.burst_time) for p in processes]
                schedule = run_algorithm(current_algorithm, processes_copy)
                process_output = process_data(schedule)
                best_algorithm = determine_best_algorithm()

def run_algorithm(algorithm, processes):
    if algorithm == "FCFS":
        return fcfs(processes)
    elif algorithm == "SJF":
        return sjf(processes)
    elif algorithm == "SRT":
        return srt(processes)
    elif algorithm == "HRRN":
        return hrrn(processes)
    elif algorithm == "RR":
        return rr(processes, 4)
    return []

def determine_best_algorithm():
    algorithms = [("FCFS", fcfs), ("SJF", sjf), ("SRT", srt), ("HRRN", hrrn), ("RR", lambda p: rr(p, 4))]
    best_algo = None
    best_avg_wait_time = float('inf')
    
    for name, algorithm in algorithms:
        processes_copy = [Process(p.pid, p.arrival_time, p.burst_time) for p in processes]
        schedule = algorithm(processes_copy)
        avg_wait_time = sum(p.waiting_time for p in processes_copy) / len(processes_copy)
        
        if avg_wait_time < best_avg_wait_time:
            best_avg_wait_time = avg_wait_time
            best_algo = name
    
    return best_algo


def process_data(schedule):
    output = []
    total_execution_time = sum(time for time, _ in schedule)
    output.append(f"Total Execution Time: {total_execution_time}")
    output.append(f"Number of Context Switches: {len(schedule)}")
    return output

ALGORITHMS = ["FCFS", "SJF", "SRT", "HRRN", "RR"]
BUTTON_HEIGHT = 50


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


current_algorithm = None
schedule = []
best_algorithm = None
process_output = []


processes = [Process(1, 0, 8), Process(2, 2, 4), Process(3, 4, 9), Process(4, 5, 5)]


cv2.namedWindow("Scheduler")
cv2.setMouseCallback("Scheduler", mouse_callback)
cv2.resizeWindow("Scheduler", WINDOW_WIDTH, WINDOW_HEIGHT)


while True:
    window_image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    draw_gui(window_image)
    cv2.imshow("Scheduler", window_image)
    key = cv2.waitKey(1)
    if key == 27:  
        break

cv2.destroyAllWindows()
