#CPU Scheduling Simulator

This code simulates a CPU scheduler and allows the user to visualize the execution of different scheduling algorithms. The implemented algorithms are:

First Come, First Served (FCFS)
Shortest Job First (SJF)
Shortest Remaining Time (SRT)
Highest Response Ratio Next (HRRN)
Round Robin (RR)
Scheduling Algorithms

First Come, First Served (FCFS): FCFS is a non-preemptive scheduling algorithm. Processes are scheduled based on their arrival time. The process that arrives first gets executed first.

Shortest Job First (SJF): SJF is a preemptive scheduling algorithm. Processes are scheduled based on their burst time. The process with the shortest burst time gets executed first.

Shortest Remaining Time (SRT): SRT is a preemptive scheduling algorithm similar to SJF. Processes are scheduled based on their remaining burst time. The process with the shortest remaining burst time gets executed first.

Highest Response Ratio Next (HRRN): HRRN is a non-preemptive scheduling algorithm. It takes into account both the waiting time and the burst time of a process. The process with the highest response ratio (calculated as the waiting time divided by the burst time) is scheduled first.

Round Robin (RR): RR is a preemptive scheduling algorithm. Each process is allocated a small time slice (quantum). A process is executed for its quantum and then preempted if it is still running. The preempted process is added back to the ready queue.

Code Functionality

The code defines a Process class to represent a process with its process ID (pid), arrival time, burst time, and other scheduling related attributes. The scheduling algorithms are implemented as separate functions: fcfs, sjf, srt, hrrn, and rr. Each function takes a list of processes as input and returns a schedule which is a list of tuples containing the time and the process ID of the process being executed at that time.

The code also includes functions to draw a graphical user interface (GUI) to display the selected algorithm, the scheduling gantt chart, and the best algorithm based on the average waiting time.

Running the Code

Save the code as a Python file (e.g., scheduler.py).
Install the required libraries: OpenCV (cv2), NumPy (np), and heapq.
Run the code from the command line: python scheduler.py
Using the GUI

The GUI displays five buttons corresponding to the implemented scheduling algorithms. Clicking on a button selects the algorithm and simulates the scheduling process. The schedule and some performance metrics are then displayed on the right side of the window.
