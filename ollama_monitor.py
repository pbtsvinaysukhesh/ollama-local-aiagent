# ollama_monitor.py

import threading
import time
import psutil
import subprocess
import logging

# Configure logging to show informational messages from the monitor
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_gpu_utilization() -> float:
    """
    Executes the nvidia-smi command to get the current GPU utilization.
    Returns 0.0 if the command fails (e.g., nvidia-smi not in PATH, or no NVIDIA GPU).
    """
    try:
        # The creationflags argument is for Windows to prevent a console window from flashing.
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        # The command returns a string like " 15 \n", so we strip and convert to float.
        utilization = float(result.stdout.strip())
        return utilization
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        # Log a warning the first time this fails, but don't spam the console.
        if not hasattr(get_gpu_utilization, "has_warned"):
            logger.warning(f"Could not get GPU utilization. nvidia-smi may not be in your PATH or you may not have an NVIDIA GPU. Error: {e}")
            get_gpu_utilization.has_warned = True
        return 0.0

class OllamaMonitor:
    """
    A monitor that finds the running Ollama process and tracks its specific
    CPU and Memory usage, as well as system-wide GPU usage, in a background thread.
    """
    def __init__(self):
        self.process = self._find_ollama_process()
        if self.process is None:
            raise RuntimeError("Ollama process not found. Please ensure the Ollama application is running.")
        
        self.cpu_samples = []
        self.mem_samples = []
        self.gpu_samples = []
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        logger.info(f"Successfully attached monitor to Ollama process (PID: {self.process.pid})")

    def _find_ollama_process(self):
        """Iterates through all running processes to find the main Ollama executable."""
        for proc in psutil.process_iter(['pid', 'name']):
            # The process name might be 'ollama.exe' on Windows or just 'ollama' on other systems.
            if 'ollama' in proc.info['name'].lower():
                return psutil.Process(proc.info['pid'])
        return None

    def _monitor(self):
        """The function that runs in the background to collect resource usage data."""
        while not self._stop_event.is_set():
            try:
                # Get CPU and Memory for the specific Ollama process.
                # The interval is important for cpu_percent to be non-blocking and accurate.
                self.cpu_samples.append(self.process.cpu_percent(interval=0.5))
                self.mem_samples.append(self.process.memory_percent())
                
                # Get system-wide GPU utilization.
                self.gpu_samples.append(get_gpu_utilization())
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                logger.warning("Ollama process was closed or access was denied during monitoring.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in the monitoring thread: {e}")
                break

    def start(self):
        """Starts the background monitoring thread."""
        self.thread.start()

    def stop(self) -> tuple[float, float, float]:
        """Stops the monitor and returns the calculated average utilization for CPU, Memory, and GPU."""
        self._stop_event.set()
        self.thread.join(timeout=1.0) # Wait a max of 1s for the thread to finish
        
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_mem = sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0
        avg_gpu = sum(self.gpu_samples) / len(self.gpu_samples) if self.gpu_samples else 0
        
        return avg_cpu, avg_mem, avg_gpu