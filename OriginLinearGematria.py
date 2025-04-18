from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHBoxLayout, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, Qt
import sys
import datetime
import csv
import itertools
import math
import random
import tempfile
import os
import platform
import shutil
import traceback
from collections import deque
import re
import asyncio
from deep_translator import GoogleTranslator
import chardet

# Try importing NumPy and Numba for GPU support
try:
    from numba import cuda, jit
    import numpy as np
    CUDA_AVAILABLE = cuda.is_available() and platform.system() == "Windows"
except ImportError:
    CUDA_AVAILABLE = False
    np = None

# Define linear_map globally
linear_map = {
    'א': 1,  'ב': 2,  'ג': 3,  'ד': 4,  'ה': 5,  'ו': 6,
    'ז': 7,  'ח': 8,  'ט': 9,  'י': 10, 'כ': 11, 'ך': 11,
    'ל': 12, 'מ': 13, 'ם': 13, 'נ': 14, 'ן': 14, 'ס': 15,
    'ע': 16, 'פ': 17, 'ף': 17, 'צ': 18, 'ץ': 18, 'ק': 19,
    'ר': 20, 'ש': 21, 'ת': 22
}
CHARSET = list(linear_map.keys())
CHARSET_SIZE = len(CHARSET)
CHAR_MAP = {c: i for i, c in enumerate(CHARSET)}

# Function to calculate base-22 prime check
BASE = 22

def get_base22_value(text):
    total = 0
    multiplier = 1
    for letter in reversed(text):
        if letter in linear_map:
            total += linear_map[letter] * multiplier
            multiplier *= BASE
    return total

def get_base10_value(text):
    total = 0
    for letter in text:
        if letter in linear_map:
            total += linear_map[letter]
    return total

def miller_rabin(n, k=5):
    """Miller-Rabin primality test for large numbers (CPU)."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False

    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def is_prime_cpu(n):
    """CPU-based primality test for small numbers."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    if n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

def is_prime(n):
    """Hybrid primality test for CPU use."""
    if n < 1000000:
        return is_prime_cpu(n)
    return miller_rabin(n)

def check_disk_space(path, min_space_mb=100):
    """Check if there is enough disk space at the given path."""
    total, used, free = shutil.disk_usage(os.path.dirname(path))
    free_mb = free // (2**20)  # Convert bytes to MB
    return free_mb >= min_space_mb

def has_valid_consecutive_chars(word):
    """Check if the word has no more than two identical consecutive characters."""
    if not word:
        return True
    count = 1
    prev_char = word[0]
    for char in word[1:]:
        if char == prev_char:
            count += 1
            if count > 2:
                return False
        else:
            count = 1
            prev_char = char
    return True

if CUDA_AVAILABLE:
    @jit(nopython=True)
    def mod_pow(base, exp, mod):
        """Modular exponentiation for CUDA compatibility."""
        result = 1
        base = base % mod
        while exp > 0:
            if exp & 1:
                result = (result * base) % mod
            base = (base * base) % mod
            exp >>= 1
        return result % mod

    @cuda.jit
    def compute_primes_kernel(indices, length, value_map, base, charset_size, fixed_positions, fixed_values, variable_positions, results):
        idx = cuda.grid(1)
        if idx >= indices.shape[0]:
            return

        try:
            val = 0
            # Add contributions from fixed positions
            for i in range(len(fixed_positions)):
                pos = fixed_positions[i]
                char_val = fixed_values[i]
                val += char_val * (base ** pos)

            # Add contributions from variable positions
            temp_idx = indices[idx]
            for i in range(len(variable_positions)):
                var_pos = variable_positions[i]
                char_idx = temp_idx % charset_size
                val += value_map[char_idx] * (base ** var_pos)
                temp_idx //= charset_size

            # Early pruning
            if val < 2 or val % 2 == 0 or val % 3 == 0:
                results[idx] = 0
                return

            # Primality test
            if val < 1000000:
                if val == 2 or val == 3:
                    results[idx] = val
                    return
                for i in range(5, int(math.sqrt(val)) + 1, 6):
                    if val % i == 0 or val % (i + 2) == 0:
                        results[idx] = 0
                        return
                results[idx] = val
            else:
                bases = [2, 3, 5, 7]
                r, d = 0, val - 1
                while d % 2 == 0:
                    r += 1
                    d //= 2
                for a in bases:
                    if a >= val:
                        continue
                    x = mod_pow(a, d, val)
                    if x == 1 or x == val - 1:
                        continue
                    for _ in range(r - 1):
                        x = (x * x) % val
                        if x == val - 1:
                            break
                    else:
                        results[idx] = 0
                        return
                results[idx] = val
        except Exception:
            results[idx] = 0

class MinerWorker(QObject):
    result_signal = pyqtSignal(list, int, str, int)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(float)
    batch_finished_signal = pyqtSignal(int)

    def __init__(self, max_len, charset, temp_file, batch_size=250000, first_char=None, last_char=None, boundary_mode=False, fixed_positions=None, variable_positions=None):
        super().__init__()
        self.charset_size = len(charset)
        self.max_len = max_len
        self.charset = charset
        self.is_running = True
        self.is_paused = False
        self.batch_size = batch_size
        self.temp_file = temp_file
        self.boundary_mode = boundary_mode
        self.first_char = first_char
        self.last_char = last_char
        self.fixed_positions = fixed_positions or []
        self.variable_positions = variable_positions or []
        self.total_combinations = (
            len(charset) ** len(self.variable_positions) if boundary_mode and self.variable_positions
            else sum(len(charset) ** length for length in range(1, max_len + 1))
        )
        self.current_batch = 0
        self.processed_in_batch = 0
        self.processed_combinations = 0
        self.entry_count = 0
        self.max_entries = batch_size
        self.entry_buffer = deque(maxlen=self.max_entries)
        self.prime_position = 0
        self.value_map = np.array([linear_map[c] for c in charset], dtype=np.int64) if CUDA_AVAILABLE else None
        self.progress_update_interval = 100

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.is_running = False
        self.is_paused = False

    def get_base22_value(self, word):
        return get_base22_value(word)

    def update_temp_file(self, new_results):
        try:
            with open(self.temp_file, 'a', encoding='utf-8') as f:
                for entry in new_results:
                    f.write(entry + '\n')
            for result in new_results:
                self.entry_buffer.append(result)
                self.entry_count += 1
                self.prime_position += 1
        except Exception as e:
            self.error_signal.emit(f"Failed to write to temp file: {str(e)}")

    def update_progress(self):
        batch_progress = (self.processed_in_batch / self.batch_size) * 100.0 if self.processed_in_batch < self.batch_size else 100.0
        self.progress_signal.emit(min(batch_progress, 100.0))

    def cpu_process_boundary_chunk(self, indices, length):
        if not self.is_running:
            return [], None, self.prime_position
        results = []
        last_word = None
        for i, idx in enumerate(indices):
            while self.is_paused and self.is_running:
                QThread.msleep(100)
            if not self.is_running:
                break
            # Initialize word with placeholders
            word = ['_'] * length
            # Set fixed characters
            for pos, char in self.fixed_positions:
                word[pos] = char
            # Fill variable positions
            temp_idx = idx
            for var_pos in self.variable_positions:
                char_idx = temp_idx % self.charset_size
                word[var_pos] = self.charset[char_idx]
                temp_idx //= self.charset_size
            word = ''.join(word)
            if len(word) != length:
                self.error_signal.emit(f"Length mismatch: Generated word '{word}' has length {len(word)}, expected {length}")
                continue
            if not has_valid_consecutive_chars(word):
                continue
            val = self.get_base22_value(word)
            if val < 2 or val % 3 == 0:
                continue
            if is_prime(val):
                result = f"{word} → {val} ✅"
                results.append(result)
                last_word = word
            if i % 1000 == 0 and i > 0:
                self.result_signal.emit([], length, word, self.prime_position)
            self.processed_in_batch += 1
            self.processed_combinations += 1
            if self.processed_in_batch % self.progress_update_interval == 0:
                self.update_progress()
        self.update_progress()
        if results:
            self.update_temp_file(results)
        return results, last_word, self.prime_position

    def gpu_process_boundary_chunk(self, indices, length):
        if not self.is_running:
            return [], None, self.prime_position
        while self.is_paused and self.is_running:
            QThread.msleep(100)
        if not self.is_running:
            return [], None, self.prime_position
        try:
            n = len(indices)
            d_indices = cuda.to_device(indices)
            d_results = cuda.device_array(n, dtype=np.int64)
            # Prepare fixed positions and values for GPU
            fixed_pos_array = np.array([pos for pos, _ in self.fixed_positions], dtype=np.int32)
            fixed_val_array = np.array([linear_map[char] for _, char in self.fixed_positions], dtype=np.int64)
            var_pos_array = np.array(self.variable_positions, dtype=np.int32)
            d_fixed_pos = cuda.to_device(fixed_pos_array)
            d_fixed_val = cuda.to_device(fixed_val_array)
            d_var_pos = cuda.to_device(var_pos_array)
            threads_per_block = 256
            blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
            compute_primes_kernel[blocks_per_grid, threads_per_block](
                d_indices, length, self.value_map, BASE, self.charset_size,
                d_fixed_pos, d_fixed_val, d_var_pos, d_results
            )
            cuda.synchronize()
            results = d_results.copy_to_host()
        except cuda.CudaAPIError as e:
            self.error_signal.emit(f"CUDA error: {str(e)}")
            return self.cpu_process_boundary_chunk(indices, length)
        except Exception as e:
            self.error_signal.emit(f"GPU processing error: {str(e)}")
            return [], None, self.prime_position

        valid_results = []
        last_word = None
        for i, val in enumerate(results):
            if val != 0:
                idx = indices[i]
                # Construct word
                word = ['_'] * length
                # Set fixed characters
                for pos, char in self.fixed_positions:
                    word[pos] = char
                # Fill variable positions
                temp_idx = idx
                for var_pos in self.variable_positions:
                    char_idx = temp_idx % self.charset_size
                    word[var_pos] = self.charset[char_idx]
                    temp_idx //= self.charset_size
                word = ''.join(word)
                if len(word) != length:
                    continue
                if not has_valid_consecutive_chars(word):
                    continue
                cpu_val = self.get_base22_value(word)
                if cpu_val != val:
                    continue
                result = f"{word} → {val} ✅"
                valid_results.append(result)
                last_word = word
            self.processed_in_batch += 1
            self.processed_combinations += 1
            if self.processed_in_batch % self.progress_update_interval == 0:
                self.update_progress()
        self.update_progress()
        if valid_results:
            self.update_temp_file(valid_results)
        return valid_results, last_word, self.prime_position

    def cpu_process_chunk(self, combos, length):
        if not self.is_running:
            return [], None, self.prime_position
        results = []
        last_word = None
        for i, combo in enumerate(combos):
            while self.is_paused and self.is_running:
                QThread.msleep(100)
            if not self.is_running:
                break
            word = ''.join(combo)
            if not has_valid_consecutive_chars(word):
                continue
            val = self.get_base22_value(word)
            if val % 2 == 0 or val % 3 == 0:
                continue
            if is_prime(val):
                result = f"{word} → {val} ✅"
                results.append(result)
                last_word = word
            if i % 1000 == 0 and i > 0:
                self.result_signal.emit([], length, '', self.prime_position)
            self.processed_in_batch += 1
            self.processed_combinations += 1
            if self.processed_in_batch % self.progress_update_interval == 0:
                self.update_progress()
        self.update_progress()
        if results:
            self.update_temp_file(results)
        return results, last_word, self.prime_position

    def run(self):
        try:
            found = 0
            batch = []
            chunk_size = 5000
            use_gpu = CUDA_AVAILABLE
            if use_gpu:
                try:
                    cuda.select_device(0)
                except Exception as e:
                    use_gpu = False
                    self.error_signal.emit(f"GPU setup failed: {str(e)}. Using CPU.")

            self.progress_signal.emit(0.0)

            if self.boundary_mode:
                combos_count = len(self.charset) ** len(self.variable_positions)
                self.processed_in_batch = 0
                for start in range(0, combos_count, chunk_size):
                    if not self.is_running:
                        break
                    end = min(start + chunk_size, combos_count)
                    indices = np.arange(start, end, dtype=np.int64)
                    if use_gpu:
                        results, last_word, position = self.gpu_process_boundary_chunk(indices, self.max_len)
                    else:
                        results, last_word, position = self.cpu_process_boundary_chunk(indices, self.max_len)
                    batch.extend(results)
                    found += len(results)
                    if self.processed_in_batch >= self.batch_size or start + chunk_size >= combos_count:
                        self.result_signal.emit(batch, self.max_len, last_word or '', position)
                        self.batch_finished_signal.emit(len(batch))
                        batch = []
                        self.processed_in_batch = 0
                        self.current_batch += 1
                        self.progress_signal.emit(0.0)
            else:
                for length in range(1, self.max_len + 1):
                    if not self.is_running:
                        break
                    combos_count = len(self.charset) ** length
                    self.processed_in_batch = 0
                    for start in range(0, combos_count, chunk_size):
                        if not self.is_running:
                            break
                        end = min(start + chunk_size, combos_count)
                        combos = [
                            tuple(
                                self.charset[(start + i) // (self.charset_size ** j) % self.charset_size]
                                for j in range(length - 1, -1, -1)
                            )
                            for i in range(end - start)
                        ]
                        results, last_word, position = self.cpu_process_chunk(combos, length)
                        batch.extend(results)
                        found += len(results)
                        if self.processed_in_batch >= self.batch_size or start + chunk_size >= combos_count:
                            self.result_signal.emit(batch, length, last_word or '', position)
                            self.batch_finished_signal.emit(len(batch))
                            batch = []
                            self.processed_in_batch = 0
                            self.current_batch += 1
                            self.progress_signal.emit(0.0)
            if batch and self.is_running:
                self.result_signal.emit(batch, self.max_len if self.boundary_mode else length, last_word or '', position)
                self.batch_finished_signal.emit(len(batch))
            if self.is_running:
                self.finished_signal.emit(found)
        except Exception as e:
            error_msg = f"Error in mining: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

class Base10Checker(QObject):
    result_signal = pyqtSignal(list)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(float)

    def __init__(self, base22_results, temp_file):
        super().__init__()
        self.base22_results = base22_results
        self.temp_file = temp_file
        self.is_running = True
        self.base10_buffer = deque()
        self.entry_count = 0

    def stop(self):
        self.is_running = False

    def update_temp_file(self, new_results):
        try:
            with open(self.temp_file, 'a', encoding='utf-8') as f:
                for entry in new_results:
                    f.write(entry + '\n')
            for result in new_results:
                self.base10_buffer.append(result)
                self.entry_count += 1
        except Exception as e:
            self.error_signal.emit(f"Failed to write to temp file: {str(e)}")

    def run(self):
        try:
            results = []
            total = len(self.base22_results)
            if total == 0:
                self.error_signal.emit("No Base-22 results to process.")
                return
            for i, entry in enumerate(self.base22_results):
                if not self.is_running:
                    break
                try:
                    word = entry.split(' → ')[0].strip()
                    if not word:
                        continue
                    base10_val = get_base10_value(word)
                    is_prime_base10 = is_prime(base10_val)
                    status = "✅Prime!✅" if is_prime_base10 else "Composite"
                    result = f"{word} → Base-10: {base10_val} ({status})"
                    results.append(result)
                    progress = ((i + 1) / total) * 100.0
                    self.progress_signal.emit(min(progress, 100.0))
                    if i % 500 == 0 or i == total - 1:
                        self.result_signal.emit(results)
                        self.update_temp_file(results)
                        results = []
                except Exception as e:
                    self.error_signal.emit(f"Error processing entry '{entry}': {str(e)}")
                    continue
            if results and self.is_running:
                self.result_signal.emit(results)
                self.update_temp_file(results)
            if self.is_running:
                self.finished_signal.emit(self.entry_count)
        except Exception as e:
            error_msg = f"Error in Base-10 checking: {str(e)}\n{traceback.format_exc()}"
            self.error_signal.emit(error_msg)

class MinerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.progress_label = QLabel("Progress: 0.00%")
        self.layout.addWidget(self.progress_label)
        self.label = QLabel("🔍 Prime Miner (Base-22)")
        self.length_label = QLabel("Max Length (characters):")
        self.length_input = QLineEdit("4")
        self.batch_size_label = QLabel("Batch Size:")
        self.batch_size_input = QLineEdit("250000")
        self.batch_size_input.setPlaceholderText("e.g., 250000")

        range_layout = QHBoxLayout()
        self.start_char_label = QLabel("Start Char:")
        self.start_char_input = QLineEdit()
        self.start_char_input.setMaxLength(1)
        self.end_char_label = QLabel("End Char:")
        self.end_char_input = QLineEdit()
        self.end_char_input.setMaxLength(1)
        self.input_text_label = QLabel("Input Text:")
        self.input_text_input = QLineEdit()
        self.char_count_label = QLabel("Char Count: 0")

        def update_fields():
            text = self.input_text_input.text().strip()
            hebrew_chars = [c for c in text if c in linear_map]
            length = len(hebrew_chars)
            self.char_count_label.setText(f"Char Count: {length}")
            self.output_area.append(f"Input: '{text}', Hebrew Chars: '{''.join(hebrew_chars)}', Length: {length}")
            if length > 0:
                boundary_text = hebrew_chars[0]
                if length > 1:
                    boundary_text += "_" * (length - 2) + hebrew_chars[-1]
                self.boundary_input.setText(boundary_text)
                self.boundary_input.repaint()
            else:
                self.boundary_input.clear()

        self.input_text_input.textChanged.connect(update_fields)

        range_layout.addWidget(self.start_char_label)
        range_layout.addWidget(self.start_char_input)
        range_layout.addWidget(self.end_char_label)
        range_layout.addWidget(self.end_char_input)
        range_layout.addWidget(self.input_text_label)
        range_layout.addWidget(self.input_text_input)
        range_layout.addWidget(self.char_count_label)

        boundary_layout = QHBoxLayout()
        self.boundary_label = QLabel("Boundary Pattern:")
        self.boundary_input = QLineEdit()
        self.boundary_input.setPlaceholderText("Use - and _ to set mining params")

        def on_focus_in(event):
            if self.boundary_input.text() == "Use - and _ to set mining params":
                self.boundary_input.clear()
            super(QLineEdit, self.boundary_input).focusInEvent(event)

        def on_focus_out(event):
            if not self.boundary_input.text():
                self.boundary_input.setText("Use - and _ to set mining params")
            super(QLineEdit, self.boundary_input).focusOutEvent(event)

        self.boundary_input.focusInEvent = on_focus_in
        self.boundary_input.focusOutEvent = on_focus_out

        boundary_layout.addWidget(self.boundary_label)
        boundary_layout.addWidget(self.boundary_input)

        button_layout = QHBoxLayout()
        self.mine_button = QPushButton("Mine Primes")
        self.boundary_mine_button = QPushButton("Mine Boundary Primes")
        self.pause_button = QPushButton("Pause Mining")
        self.stop_button = QPushButton("Stop Mining")
        self.check_base10_button = QPushButton("Check Base 10")
        button_layout.addWidget(self.mine_button)
        button_layout.addWidget(self.boundary_mine_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.check_base10_button)

        save_layout = QHBoxLayout()
        self.save_button = QPushButton("💾 Save Base-22 Results")
        self.save_base10_button = QPushButton("💾 Save Base-10 Results")
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.save_base10_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.length_label)
        self.layout.addWidget(self.length_input)
        self.layout.addWidget(self.batch_size_label)
        self.layout.addWidget(self.batch_size_input)
        self.layout.addLayout(range_layout)
        self.layout.addLayout(boundary_layout)
        self.layout.addLayout(button_layout)
        self.layout.addLayout(save_layout)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.output_area)
        self.setLayout(self.layout)

        self.mine_button.clicked.connect(self.start_mining)
        self.boundary_mine_button.clicked.connect(self.start_boundary_mining)
        self.pause_button.clicked.connect(self.toggle_pause_mining)
        self.stop_button.clicked.connect(self.stop_mining)
        self.check_base10_button.clicked.connect(self.check_base10)
        self.save_button.clicked.connect(self.save_results)
        self.save_base10_button.clicked.connect(self.save_base10_results)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.check_base10_button.setEnabled(False)
        self.save_base10_button.setEnabled(False)

        self.thread = None
        self.worker = None
        self.base10_thread = None
        self.base10_worker = None
        self.temp_file = None
        self.base10_temp_file = None
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.flush_progress)
        self.pending_progress = 0.0
        self.is_paused = False

    def extract_boundary_params(self, boundary_text):
        """Parse boundary pattern to extract fixed characters, their positions, and variable positions."""
        if boundary_text == "Use - and _ to set mining params":
            return None, None, 0, [], []

        fixed_positions = []
        variable_positions = []
        for i, char in enumerate(boundary_text):
            if char in linear_map:
                fixed_positions.append((i, char))
            elif char in ('-', '_'):
                variable_positions.append(i)
            else:
                return None, None, 0, [], []

        if not fixed_positions:
            return None, None, 0, [], []

        first_char = fixed_positions[0][1] if fixed_positions else None
        last_char = fixed_positions[-1][1] if fixed_positions else None
        total_length = len(boundary_text)

        return first_char, last_char, total_length, fixed_positions, variable_positions

    def start_mining(self):
        try:
            max_len = int(self.length_input.text())
            if max_len <= 0:
                self.output_area.setText("Please enter a positive number for length.")
                return
            if max_len > 10:
                self.output_area.setText("Max length cannot exceed 10 to prevent crashes.")
                return
        except ValueError:
            self.output_area.setText("Invalid length.")
            return

        try:
            batch_size = int(self.batch_size_input.text())
            if batch_size <= 0:
                self.output_area.setText("Batch size must be a positive number.")
                return
            if batch_size > 500000:
                self.output_area.setText("Batch size cannot exceed 500,000 to prevent crashes.")
                return
        except ValueError:
            self.output_area.setText("Invalid batch size.")
            return

        start_char = self.start_char_input.text().strip()
        end_char = self.end_char_input.text().strip()
        filtered_charset = CHARSET

        if start_char and end_char:
            try:
                s_idx = CHARSET.index(start_char)
                e_idx = CHARSET.index(end_char)
                if s_idx <= e_idx:
                    filtered_charset = CHARSET[s_idx:e_idx + 1]
                else:
                    filtered_charset = CHARSET[e_idx:s_idx + 1]
            except ValueError:
                self.output_area.setText(f"Invalid characters. '{start_char}' or '{end_char}' not found.")
                return

        if max_len > 5:
            total_combinations = sum(len(filtered_charset) ** length for length in range(1, max_len + 1))
            reply = QMessageBox.question(
                self, "Warning",
                f"Length {max_len} will process {total_combinations:,} combinations. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        if not check_disk_space(tempfile.gettempdir()):
            self.output_area.setText("Insufficient disk space. Please free up at least 100 MB.")
            return

        self.cleanup_thread()
        self.output_area.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0.00%")
        self.pending_progress = 0.0
        self.mine_button.setEnabled(False)
        self.boundary_mine_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.check_base10_button.setEnabled(False)
        self.save_base10_button.setEnabled(False)
        self.pause_button.setText("Pause Mining")
        self.is_paused = False

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False).name

        self.thread = QThread()
        self.worker = MinerWorker(max_len, filtered_charset, self.temp_file, batch_size=batch_size)

        self.worker.moveToThread(self.thread)
        self.worker.result_signal.connect(self.append_batch)
        self.worker.batch_finished_signal.connect(self.on_batch_finished)
        self.worker.finished_signal.connect(self.on_mining_finished)
        self.worker.error_signal.connect(self.on_mining_error)
        self.worker.progress_signal.connect(self.queue_progress)
        self.thread.started.connect(self.worker.run)

        self.output_area.append("Starting mining...")
        self.progress_timer.start(100)
        self.thread.start()
        QApplication.processEvents()

    def start_boundary_mining(self):
        boundary_text = self.boundary_input.text().strip()
        first_char, last_char, total_length, fixed_positions, variable_positions = self.extract_boundary_params(boundary_text)

        if not fixed_positions or total_length < 1:
            self.output_area.setText("Please provide a valid boundary pattern (e.g., 'ה__ת' or 'את-----יה').")
            return

        try:
            batch_size = int(self.batch_size_input.text())
            if batch_size <= 0:
                self.output_area.setText("Batch size must be a positive number.")
                return
            if batch_size > 500000:
                self.output_area.setText("Batch size cannot exceed 500,000 to prevent crashes.")
                return
        except ValueError:
            self.output_area.setText("Invalid batch size.")
            return

        if not all(char in linear_map for _, char in fixed_positions):
            self.output_area.setText("All fixed characters must be valid Hebrew letters.")
            return

        if total_length > 10:
            middle_combos = len(CHARSET) ** len(variable_positions)
            reply = QMessageBox.question(
                self, "Warning",
                f"Length {total_length} with {len(variable_positions)} variable positions will process {middle_combos:,} combinations. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        if not check_disk_space(tempfile.gettempdir()):
            self.output_area.setText("Insufficient disk space. Please free up at least 100 MB.")
            return

        self.cleanup_thread()
        self.output_area.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0.00%")
        self.pending_progress = 0.0
        self.mine_button.setEnabled(False)
        self.boundary_mine_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.check_base10_button.setEnabled(False)
        self.save_base10_button.setEnabled(False)
        self.pause_button.setText("Pause Mining")
        self.is_paused = False

        self.temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False).name

        self.thread = QThread()
        self.worker = MinerWorker(
            total_length, CHARSET, self.temp_file,
            batch_size=batch_size, first_char=first_char, last_char=last_char,
            boundary_mode=True, fixed_positions=fixed_positions, variable_positions=variable_positions
        )

        self.worker.moveToThread(self.thread)
        self.worker.result_signal.connect(self.append_batch)
        self.worker.batch_finished_signal.connect(self.on_batch_finished)
        self.worker.finished_signal.connect(self.on_mining_finished)
        self.worker.error_signal.connect(self.on_mining_error)
        self.worker.progress_signal.connect(self.queue_progress)
        self.thread.started.connect(self.worker.run)

        self.output_area.append("Starting boundary mining...")
        self.progress_timer.start(100)
        self.thread.start()
        QApplication.processEvents()

    def toggle_pause_mining(self):
        if self.worker:
            if self.is_paused:
                self.worker.resume()
                self.pause_button.setText("Pause Mining")
                self.output_area.append("▶️ Mining resumed")
                self.save_button.setEnabled(False)
                self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
                self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
            else:
                self.worker.pause()
                self.pause_button.setText("Resume Mining")
                self.output_area.append("⏸️ Mining paused")
                self.save_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
                self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
                self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
            self.is_paused = not self.is_paused
        QApplication.processEvents()

    def stop_mining(self):
        if self.worker:
            self.worker.stop()
            self.cleanup_thread()
        self.output_area.append("🛑 Mining stopped")
        self.save_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
        QApplication.processEvents()

    def check_base10(self):
        if not self.worker or not self.worker.entry_buffer:
            self.output_area.append("❌ No Base-22 results to check for Base-10 primality.")
            return

        try:
            base22_results = list(self.worker.entry_buffer)
            if not base22_results:
                self.output_area.append("❌ Base-22 results buffer is empty.")
                return

            self.cleanup_base10_thread()
            self.progress_bar.setValue(0)
            self.progress_label.setText("Base-10 Progress: 0.00%")
            self.check_base10_button.setEnabled(False)
            self.save_base10_button.setEnabled(False)

            self.base10_temp_file = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.txt', delete=False).name

            self.base10_thread = QThread()
            self.base10_worker = Base10Checker(base22_results, self.base10_temp_file)

            self.base10_worker.moveToThread(self.base10_thread)
            self.base10_worker.result_signal.connect(self.append_base10_results)
            self.base10_worker.finished_signal.connect(self.on_base10_finished)
            self.base10_worker.error_signal.connect(self.on_base10_error)
            self.base10_worker.progress_signal.connect(self.queue_progress)
            self.base10_thread.started.connect(self.base10_worker.run)

            self.output_area.append("Starting Base-10 primality check...")
            self.progress_timer.start(100)
            self.base10_thread.start()
            QApplication.processEvents()
        except Exception as e:
            self.output_area.append(f"❌ Failed to start Base-10 check: {str(e)}\n{traceback.format_exc()}")
            self.check_base10_button.setEnabled(True)
            self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
            QApplication.processEvents()

    def queue_progress(self, progress):
        self.pending_progress = progress

    def flush_progress(self):
        if self.pending_progress is not None:
            if self.base10_thread and self.base10_thread.isRunning():
                self.progress_label.setText(f"Base-10 Progress: {self.pending_progress:.2f}%")
            else:
                self.progress_label.setText(f"Progress: {self.pending_progress:.2f}%")
            self.progress_bar.setValue(int(self.pending_progress))
            self.progress_bar.repaint()
            QApplication.processEvents()

    def append_batch(self, batch, length, last_word, position):
        if batch:
            if len(batch) > 100:
                self.output_area.append(f"Found {len(batch)} Base-22 primes (most recent: {last_word or 'none'}, {length} chars, prime #{position})")
            else:
                for result in batch:
                    self.output_area.append(result)
            self.save_button.setEnabled(True)
            self.check_base10_button.setEnabled(True)
        QApplication.processEvents()

    def append_base10_results(self, results):
        if results:
            if len(results) > 100:
                self.output_area.append(f"Processed {len(results)} Base-10 results")
            else:
                for result in results:
                    self.output_area.append(result)
        self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
        QApplication.processEvents()

    def on_batch_finished(self, batch_size):
        self.progress_label.setText("Progress: 100.00%")
        self.progress_bar.setValue(100)
        self.progress_bar.repaint()
        self.output_area.append(f"Base-22 batch completed with {batch_size} primes found.")
        self.save_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
        QApplication.processEvents()

    def on_mining_finished(self, found):
        self.output_area.append(f"\n🧱 Total Base-22 primes found: {found}")
        self.progress_label.setText("Progress: 100.00%")
        self.progress_bar.setValue(100)
        self.cleanup_thread()
        self.save_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
        QApplication.processEvents()

    def on_mining_error(self, error):
        self.output_area.append(f"❌ Base-22 mining error: {error}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0.00%")
        self.cleanup_thread()
        self.save_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        self.save_base10_button.setEnabled(self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0)
        QApplication.processEvents()

    def on_base10_finished(self, count):
        has_base10_results = self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0
        self.output_area.append(f"\n✅ Base-10 primality check completed: {count} results processed")
        self.progress_label.setText("Base-10 Progress: 100.00%")
        self.progress_bar.setValue(100)
        self.save_base10_button.setEnabled(has_base10_results)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        QApplication.processEvents()

    def on_base10_error(self, error):
        has_base10_results = self.base10_worker is not None and len(self.base10_worker.base10_buffer) > 0
        self.output_area.append(f"❌ Base-10 checking error: {error}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Base-10 Progress: 0.00%")
        self.save_base10_button.setEnabled(has_base10_results)
        self.check_base10_button.setEnabled(self.worker and len(self.worker.entry_buffer) > 0)
        QApplication.processEvents()

    def estimate_file_size(self, entries):
        total_bytes = sum(len((entry + '\n').encode('utf-8')) for entry in entries)
        if total_bytes < 1024:
            return f"{total_bytes} bytes"
        elif total_bytes < 1024 * 1024:
            return f"{total_bytes / 1024:.2f} KB"
        else:
            return f"{total_bytes / (1024 * 1024):.2f} MB"

    def save_results(self):
        if not self.worker or not self.worker.entry_buffer:
            self.output_area.append("❌ No Base-22 results to save.")
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"prime_miner_base22_results_{timestamp}.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Base-22 Results",
            default_filename,
            "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            self.output_area.append("Base-22 save cancelled.")
            return
        try:
            file_size_str = self.estimate_file_size(self.worker.entry_buffer)
            self.output_area.append(f"Attempting to save Base-22 results to: {filename}")
            if not check_disk_space(filename):
                self.output_area.append("❌ Insufficient disk space. Please free up at least 100 MB.")
                return
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.worker.entry_buffer:
                    f.write(entry + '\n')
            if os.path.getsize(filename) == 0:
                self.output_area.append("❌ No Base-22 results to save. Output file is empty.")
                os.remove(filename)
                return
            self.output_area.append(f"✅ Base-22 results saved as {filename} (Estimated size: {file_size_str})")
        except Exception as e:
            self.output_area.append(f"❌ Error saving Base-22 results: {str(e)}")
        QApplication.processEvents()

    def save_base10_results(self):
        if not self.base10_worker or not self.base10_worker.base10_buffer:
            self.output_area.append("❌ No Base-10 results to save.")
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"prime_miner_base10_results_{timestamp}.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Base-10 Results",
            default_filename,
            "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            self.output_area.append("Base-10 save cancelled.")
            return
        try:
            file_size_str = self.estimate_file_size(self.base10_worker.base10_buffer)
            self.output_area.append(f"Attempting to save Base-10 results to: {filename}")
            if not check_disk_space(filename):
                self.output_area.append("❌ Insufficient disk space. Please free up at least 100 MB.")
                return
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.base10_worker.base10_buffer:
                    f.write(entry + '\n')
            if os.path.getsize(filename) == 0:
                self.output_area.append("❌ No Base-10 results to save. Output file is empty.")
                os.remove(filename)
                return
            self.output_area.append(f"✅ Base-10 results saved as {filename} (Estimated size: {file_size_str})")
        except Exception as e:
            self.output_area.append(f"❌ Error saving Base-10 results: {str(e)}")
        QApplication.processEvents()

    def cleanup_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
        self.progress_timer.stop()
        self.mine_button.setEnabled(True)
        self.boundary_mine_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.pause_button.setText("Pause Mining")
        self.is_paused = False
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.unlink(self.temp_file)
            except Exception as e:
                self.output_area.append(f"Warning: Failed to delete temp file {self.temp_file}: {str(e)}")
            self.temp_file = None
        self.cleanup_base10_thread(reason="Starting new mining operation")

    def cleanup_base10_thread(self, reason="Unknown"):
        if self.base10_worker and len(self.base10_worker.base10_buffer) > 0:
            self.output_area.append(f"Warning: Cleaning up Base-10 thread with {len(self.base10_worker.base10_buffer)} unsaved results. Reason: {reason}")
        if self.base10_thread and self.base10_thread.isRunning():
            if self.base10_worker:
                self.base10_worker.stop()
            self.base10_thread.quit()
            self.base10_thread.wait()
        self.base10_thread = None
        self.base10_worker = None
        if self.base10_temp_file and os.path.exists(self.base10_temp_file):
            try:
                os.unlink(self.base10_temp_file)
            except Exception as e:
                self.output_area.append(f"Warning: Failed to delete temp file {self.base10_temp_file}: {str(e)}")
            self.base10_temp_file = None
        self.save_base10_button.setEnabled(False)
        QApplication.processEvents()

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        if self.base10_worker:
            self.base10_worker.stop()
        self.cleanup_thread()
        super().closeEvent(event)

class GematriaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.entries = []

        self.input_label = QLabel("Enter Hebrew text:")
        self.input_field = QLineEdit()
        self.calculate_button = QPushButton("Calculate")
        self.result_label = QLabel("Result: —")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.save_button = QPushButton("💾 Save Logs to File")

        self.keyboard_layout = QVBoxLayout()
        self.add_keyboard()

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addLayout(self.keyboard_layout)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.log_area)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.calculate_button.clicked.connect(self.calculate_gematria)
        self.save_button.clicked.connect(self.save_logs)
        self.input_field.installEventFilter(self)

    def add_keyboard(self):
        rows = QVBoxLayout()
        row = QHBoxLayout()
        for i, letter in enumerate(linear_map.keys()):
            button = QPushButton(letter)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _, l=letter: self.input_field.insert(l))
            row.addWidget(button)
            if (i + 1) % 10 == 0:
                rows.addLayout(row)
                row = QHBoxLayout()
        rows.addLayout(row)

        space_row = QHBoxLayout()
        space_row.addStretch()
        space_button = QPushButton("␣")
        space_button.setFixedSize(150, 40)
        space_button.clicked.connect(lambda: self.input_field.insert(" "))
        space_row.addWidget(space_button)
        space_row.addStretch()
        rows.addLayout(space_row)

        self.keyboard_layout = rows

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent, Qt
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self.input_field.insert(" ")
                return True
        return super().eventFilter(obj, event)

    def calculate_gematria(self):
        text = self.input_field.text().strip()
        if not text:
            self.result_label.setText("Result: (Empty input)")
            return

        base22_val = get_base22_value(text)
        prime_status = "✅Prime!✅" if is_prime(base22_val) else "Composite"
        result_display = f"Base-22: {base22_val} ({prime_status})"

        self.result_label.setText(f"Result: {result_display}")
        self.log_area.append(f"{text} → {base22_val} ({prime_status})")
        self.entries.append((text, base22_val, prime_status))

    def save_logs(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_filename = f"gematria_base22_log_{timestamp}.txt"
        csv_filename = f"gematria_base22_log_{timestamp}.csv"

        try:
            with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                for entry in self.entries:
                    txt_file.write(f"{entry[0]} → {entry[1]} ({entry[2]})\n")

            with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Input', 'Base-22 Value', 'Prime?'])
                writer.writerows(self.entries)

            self.result_label.setText(f"✅ Logs saved as {txt_filename} + {csv_filename}")
        except Exception as e:
            self.result_label.setText(f"❌ Error saving: {e}")

class Base10GematriaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.entries = []

        self.input_label = QLabel("Enter Hebrew text:")
        self.input_field = QLineEdit()
        self.calculate_button = QPushButton("Calculate")
        self.result_label = QLabel("Result: —")
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.save_button = QPushButton("💾 Save Logs to File")

        self.keyboard_layout = QVBoxLayout()
        self.add_keyboard()

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addLayout(self.keyboard_layout)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.log_area)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.calculate_button.clicked.connect(self.calculate_gematria)
        self.save_button.clicked.connect(self.save_logs)
        self.input_field.installEventFilter(self)

    def add_keyboard(self):
        rows = QVBoxLayout()
        row = QHBoxLayout()
        for i, letter in enumerate(linear_map.keys()):
            button = QPushButton(letter)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _, l=letter: self.input_field.insert(l))
            row.addWidget(button)
            if (i + 1) % 10 == 0:
                rows.addLayout(row)
                row = QHBoxLayout()
        rows.addLayout(row)

        space_row = QHBoxLayout()
        space_row.addStretch()
        space_button = QPushButton("␣")
        space_button.setFixedSize(150, 40)
        space_button.clicked.connect(lambda: self.input_field.insert(" "))
        space_row.addWidget(space_button)
        space_row.addStretch()
        rows.addLayout(space_row)

        self.keyboard_layout = rows

    def eventFilter(self, obj, event):
        from PyQt5.QtCore import QEvent, Qt
        if obj == self.input_field and event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Space:
                self.input_field.insert(" ")
                return True
        return super().eventFilter(obj, event)

    def calculate_gematria(self):
        text = self.input_field.text().strip()
        if not text:
            self.result_label.setText("Result: (Empty input)")
            return

        base10_val = get_base10_value(text)
        prime_status = "✅Prime!✅" if is_prime(base10_val) else "Composite"
        result_display = f"Base-10: {base10_val} ({prime_status})"

        self.result_label.setText(f"Result: {result_display}")
        self.log_area.append(f"{text} → {base10_val} ({prime_status})")
        self.entries.append((text, base10_val, prime_status))

    def save_logs(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_filename = f"gematria_base10_log_{timestamp}.txt"
        csv_filename = f"gematria_base10_log_{timestamp}.csv"

        try:
            with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                for entry in self.entries:
                    txt_file.write(f"{entry[0]} → {entry[1]} ({entry[2]})\n")

            with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Input', 'Base-10 Value', 'Prime?'])
                writer.writerows(self.entries)

            self.result_label.setText(f"✅ Logs saved as {txt_filename} + {csv_filename}")
        except Exception as e:
            self.result_label.setText(f"❌ Error saving: {e}")

class InfoTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("📚 Aleph-Bet Linear Gematria Table")
        layout.addWidget(label)

        table = QTableWidget()
        table.setRowCount(len(linear_map))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Letter", "Value"])

        for i, (letter, value) in enumerate(linear_map.items()):
            table.setItem(i, 0, QTableWidgetItem(letter))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        self.setLayout(layout)

class KeyboardTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.textbox = QTextEdit()
        self.textbox.setFont(QFont("Arial", 16))
        self.layout.addWidget(self.textbox)

        self.keyboard_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.letters = list(linear_map.keys())

        for i, letter in enumerate(self.letters):
            button = QPushButton(letter)
            button.setFixedSize(40, 40)
            button.clicked.connect(lambda _, l=letter: self.textbox.insertPlainText(l))
            row.addWidget(button)
            if (i + 1) % 10 == 0:
                self.keyboard_layout.addLayout(row)
                row = QHBoxLayout()
        self.keyboard_layout.addLayout(row)
        self.layout.addLayout(self.keyboard_layout)

        self.copy_button = QPushButton("📋 Copy Text")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.layout.addWidget(self.copy_button)

        self.setLayout(self.layout)

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.textbox.toPlainText())

class GoogleTranslateTab(QWidget):
    def __init__(self):
        super().__init__()
        self.translations = []
        self.init_ui()
        self.thread = None
        self.worker = None

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.hebrew_label = QLabel("Hebrew Input:")
        self.hebrew_input = QTextEdit()
        self.hebrew_input.setAcceptRichText(False)
        self.hebrew_input.setPlaceholderText("Enter or paste Hebrew text here...")
        self.layout.addWidget(self.hebrew_label)
        self.layout.addWidget(self.hebrew_input)

        self.english_label = QLabel("English Translation:")
        self.english_output = QTextEdit()
        self.english_output.setReadOnly(True)
        self.english_output.setPlaceholderText("Translation will appear here...")
        self.layout.addWidget(self.english_label)
        self.layout.addWidget(self.english_output)

        self.file_layout = QHBoxLayout()
        self.file_label = QLabel("Selected File: None")
        self.select_file_button = QPushButton("📁 Select .txt File")
        self.select_file_button.clicked.connect(self.select_file)
        self.file_layout.addWidget(self.select_file_button)
        self.file_layout.addWidget(self.file_label)
        self.layout.addLayout(self.file_layout)

        self.button_layout = QHBoxLayout()
        self.translate_button = QPushButton("Translate Text")
        self.translate_button.clicked.connect(self.translate_text)
        self.process_file_button = QPushButton("Process File")
        self.process_file_button.clicked.connect(self.process_file)
        self.process_file_button.setEnabled(False)
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.save_button = QPushButton("💾 Save Translation Log")
        self.save_button.clicked.connect(self.save_log)
        self.save_button.setEnabled(False)
        self.button_layout.addWidget(self.translate_button)
        self.button_layout.addWidget(self.process_file_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.save_button)
        self.layout.addLayout(self.button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("Progress: 0.00%")
        self.layout.addWidget(self.progress_label)
        self.layout.addWidget(self.progress_bar)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Translation logs will appear here...")
        self.layout.addWidget(self.log_area)

        self.setLayout(self.layout)

    def select_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Text File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if filename:
            self.file_label.setText(f"Selected File: {os.path.basename(filename)}")
            self.file_path = filename
            self.process_file_button.setEnabled(True)
            self.log_area.append(f"Selected file: {filename}")
        else:
            self.file_label.setText("Selected File: None")
            self.process_file_button.setEnabled(False)

    async def translate_text_async(self, text):
        """Asynchronously translate text from Hebrew to English."""
        translator = GoogleTranslator(source='iw', target='en')
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: translator.translate(text))
        except Exception as e:
            return f"Translation error: {str(e)}"

    def translate_text(self):
        """Translate text from Hebrew input to English output."""
        hebrew_text = self.hebrew_input.toPlainText().strip()
        if not hebrew_text:
            self.english_output.setPlainText("Please enter Hebrew text.")
            return

        try:
            translation = asyncio.run(self.translate_text_async(hebrew_text))
            self.english_output.setPlainText(translation)
            self.translations.append((hebrew_text, translation))
            self.log_area.append(f"Hebrew: {hebrew_text}\nEnglish: {translation}\n")
            self.save_button.setEnabled(True)
        except Exception as e:
            self.english_output.setPlainText(f"Translation error: {str(e)}")
            self.log_area.append(f"Hebrew: {hebrew_text}\nError: {str(e)}\n")

    def process_file(self):
        if not hasattr(self, 'file_path') or not os.path.exists(self.file_path):
            self.log_area.append("❌ Please select a valid .txt file.")
            return

        self.cleanup_thread()
        self.translations = []
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0.00%")
        self.process_file_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.log_area.append("Starting file processing...")

        self.thread = QThread()
        self.worker = TranslatorWorker(self.file_path)
        self.worker.moveToThread(self.thread)
        self.worker.result_signal.connect(self.append_translation)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.error_signal.connect(self.on_processing_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.cleanup_thread()
            self.log_area.append("🛑 File processing stopped.")
            self.save_button.setEnabled(len(self.translations) > 0)

    def append_translation(self, hebrew, english):
        self.log_area.append(f"Hebrew: {hebrew}\nEnglish: {english}\n")
        self.translations.append((hebrew, english))
        self.save_button.setEnabled(True)

    def update_progress(self, progress):
        self.progress_bar.setValue(int(progress))
        self.progress_label.setText(f"Progress: {progress:.2f}%")

    def on_processing_finished(self, count):
        self.log_area.append(f"✅ File processing completed: {count} translations.")
        self.progress_bar.setValue(100)
        self.progress_label.setText("Progress: 100.00%")
        self.cleanup_thread()
        self.save_button.setEnabled(len(self.translations) > 0)

    def on_processing_error(self, error):
        self.log_area.append(f"❌ Error: {error}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Progress: 0.00%")
        self.cleanup_thread()
        self.save_button.setEnabled(len(self.translations) > 0)

    def save_log(self):
        if not self.translations:
            self.log_area.append("❌ No translations to save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"translation_log_{timestamp}.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Translation Log",
            default_filename,
            "Text Files (*.txt);;All Files (*)"
        )
        if not filename:
            self.log_area.append("Save cancelled.")
            return

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for hebrew, english in self.translations:
                    f.write(f"Hebrew: {hebrew}\nEnglish: {english}\n\n")
            file_size = os.path.getsize(filename)
            size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.2f} MB"
            self.log_area.append(f"✅ Log saved as {filename} (Size: {size_str})")
        except Exception as e:
            self.log_area.append(f"❌ Error saving log: {str(e)}")

    def cleanup_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None
        self.process_file_button.setEnabled(hasattr(self, 'file_path') and os.path.exists(self.file_path))
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        self.cleanup_thread()
        super().closeEvent(event)

class TranslatorWorker(QObject):
    result_signal = pyqtSignal(str, str)
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.is_running = True
        self.translator = GoogleTranslator(source='iw', target='en')
        self.translations = []

    def stop(self):
        self.is_running = False

    def detect_encoding(self, file_path):
        """Detect the file encoding using chardet."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']

    def extract_hebrew(self, line):
        """Extract Hebrew characters from a line, assuming line contains ✅."""
        hebrew_chars = re.findall(r'[\u0590-\u05FF]+', line)
        return ''.join(hebrew_chars).strip()

    async def translate_text(self, text):
        """Asynchronously translate text from Hebrew to English."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.translator.translate(text))
        except Exception as e:
            return f"Translation error: {str(e)}"

    def run(self):
        try:
            encoding = self.detect_encoding(self.file_path)
            with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                lines = f.readlines()

            total_lines = len(lines)
            if total_lines == 0:
                self.error_signal.emit("File is empty.")
                return

            processed = 0
            for i, line in enumerate(lines):
                if not self.is_running:
                    break
                if '✅' in line:
                    hebrew_text = self.extract_hebrew(line)
                    if hebrew_text:
                        try:
                            translation = asyncio.run(self.translate_text(hebrew_text))
                            if "Translation error" not in translation:
                                self.translations.append((hebrew_text, translation))
                                self.result_signal.emit(hebrew_text, translation)
                            else:
                                self.result_signal.emit(hebrew_text, translation)
                        except Exception as e:
                            self.result_signal.emit(hebrew_text, f"Translation error: {str(e)}")
                processed += 1
                progress = (processed / total_lines) * 100.0
                self.progress_signal.emit(progress)

            if self.is_running:
                self.finished_signal.emit(len(self.translations))
        except Exception as e:
            self.error_signal.emit(f"Error processing file: {str(e)}")

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧮 Hebrew Gematria Tool")
        self.setGeometry(200, 200, 500, 600)

        self.gematria_tab = GematriaTab()
        self.base10_gematria_tab = Base10GematriaTab()
        self.info_tab = InfoTab()
        self.keyboard_tab = KeyboardTab()
        self.miner_tab = MinerTab()
        self.translate_tab = GoogleTranslateTab()

        self.addTab(self.gematria_tab, "Base-22 Calculator")
        self.addTab(self.base10_gematria_tab, "Base-10 Calculator")
        self.addTab(self.info_tab, "Info")
        self.addTab(self.keyboard_tab, "Hebrew Keyboard")
        self.addTab(self.miner_tab, "Prime Miner")
        self.addTab(self.translate_tab, "Google Translate")

        self.adjustTabBarWidth()

    def adjustTabBarWidth(self):
        tab_bar = self.tabBar()
        total_width = 0
        for i in range(tab_bar.count()):
            total_width += tab_bar.tabRect(i).width()
        total_width += 20
        self.setMinimumWidth(total_width + 100)
        tab_bar.setMinimumWidth(total_width)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())