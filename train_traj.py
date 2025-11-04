import threading
import queue
from tqdm import tqdm
from functools import partial

def worker(task_queue, results, progress_bar, input_deck, report_dir):
    """Thread worker that processes cases from the queue."""
    while True:
        case = task_queue.get()
        if case is None:
            task_queue.task_done()
            break
        try:
            result = simulate_trajectory(case, input_deck, report_dir)
            results.append(result)
        except Exception as e:
            print(f"Error processing case {case}: {e}")
        finally:
            task_queue.task_done()
            progress_bar.update(1)

def run_threaded(cases, input_deck, report_dir, num_threads=4):
    """Main function that runs the queue-based threading workflow."""
    task_queue = queue.Queue()
    results = []
    progress_bar = tqdm(total=len(cases), desc="Processing Samples")

    # Spawn worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(
            target=worker, 
            args=(task_queue, results, progress_bar, input_deck, report_dir),
            daemon=True
        )
        t.start()
        threads.append(t)

    # Fill queue with work
    for case in cases:
        task_queue.put(case)

    # Wait until all tasks are done
    task_queue.join()

    # Stop workers
    for _ in range(num_threads):
        task_queue.put(None)
    for t in threads:
        t.join()

    progress_bar.close()
    return results


# Example usage
cases = generate_design_space(variables, num_samples)
results = run_threaded(cases, input_deck, report_dir, num_threads=num_processors)
