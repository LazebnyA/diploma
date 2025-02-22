import io
import sys


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()


# Logger decorator that captures stdout/stderr and writes logs to a file after training
def logger_decorator(version: str):
    def inner(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            log_capture = io.StringIO()
            sys.stdout = Tee(original_stdout, log_capture)
            sys.stderr = Tee(original_stderr, log_capture)
            result = None
            try:
                result = func(*args, **kwargs)
            except KeyboardInterrupt:
                print("Training interrupted by user.")
            finally:
                # Restore original output streams
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                # Retrieve completed_epochs from the function's return value, if available
                completed_epochs = 0
                if result is not None and isinstance(result, dict) and "completed_epochs" in result:
                    completed_epochs = result["completed_epochs"]
                base_filename = f"cnn_lstm_ctc_handwritten_v{version}_{completed_epochs + 1}ep"
                log_filename = f"{base_filename}.txt"
                with open(log_filename, "w") as f:
                    f.write(log_capture.getvalue())
                print(f"Console output logged to {log_filename}")
            return result

        return wrapper

    return inner
