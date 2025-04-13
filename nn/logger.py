import io
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# Logger decorator that captures stdout/stderr and writes logs to a file after training
def logger_model_training(version: str, additional: str = None):
    def inner(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            log_capture = io.StringIO()
            sys.stdout = Tee(original_stdout, log_capture)
            sys.stderr = Tee(original_stderr, log_capture)
            result = None
            try:
                result = func(version, additional, *args, **kwargs)
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
                base_filename = f"cnn_lstm_ctc_handwritten_v{version}_{completed_epochs + 1}ep_{additional}" \
                    if additional else f"cnn_lstm_ctc_handwritten_v{version}_{completed_epochs + 1}ep"
                log_filename = f"{base_filename}.txt"
                with open(log_filename, "w", encoding='utf-8') as f:
                    f.write(log_capture.getvalue())
                print(f"Console output logged to {log_filename}")
            return result

        return wrapper

    return inner


def evaluation_logger(model_description: str = None):
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
                base_filename = f"{model_description}_evaluation_logs" if not None else "evaluation_logs"
                log_filename = f"{base_filename}.txt"
                with open(log_filename, "w", encoding='utf-8') as f:
                    f.write(log_capture.getvalue())
                print(f"Console output logged to {log_filename}")
            return result

        return wrapper

    return inner


def logger_hyperparameters_tuning(model_description):
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
                sys.stdout = original_stdout
                sys.stderr = original_stderr

                # Filter captured output
                captured_output = log_capture.getvalue()
                filtered_lines = [
                    line for line in captured_output.splitlines()
                    if not line.strip().startswith("Training Epoch") and "%|" not in line and not line.strip().startswith('\r')
                ]
                filtered_output = "\n".join(filtered_lines)

                # Save filtered output
                base_filename = f"{model_description}_hyperparams_tuning_logs" if model_description else "hyperparams_tuning_logs"
                log_filename = f"{base_filename}.txt"
                with open(log_filename, "w", encoding='utf-8') as f:
                    f.write(filtered_output)

                print(f"Filtered log saved to {log_filename}")

            return result

        return wrapper

    return inner