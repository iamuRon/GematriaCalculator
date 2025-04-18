from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHBoxLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QThread, pyqtSignal, QObject
import sys
import datetime
import csv
import itertools

# Define linear_map globally
linear_map = {
    'א': 1,  'ב': 2,  'ג': 3,  'ד': 4,  'ה': 5,  'ו': 6,
    'ז': 7,  'ח': 8,  'ט': 9,  'י': 10, 'כ': 11, 'ך': 11,
    'ל': 12, 'מ': 13, 'ם': 13, 'נ': 14, 'ן': 14, 'ס': 15,
    'ע': 16, 'פ': 17, 'ף': 17, 'צ': 18, 'ץ': 18, 'ק': 19,
    'ר': 20, 'ש': 21, 'ת': 22
}

BASE = 22

def get_linear_gematria(text):
    total = 0
    for letter in text:
        if letter in linear_map:
            total += linear_map[letter]
    return total

def get_base22_value(text):
    total = 0
    multiplier = 1
    for letter in reversed(text):
        if letter in linear_map:
            total += linear_map[letter] * multiplier
            multiplier *= BASE
    return total

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5)+1):
        if n % i == 0:
            return False
    return True

class Worker(QObject):
    update = pyqtSignal(str)
    done = pyqtSignal(str)

    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.func(*self.args, self.update, self.done)

def run_prime_miner(max_len, update_cb, done_cb):
    charset = list(linear_map.keys())
    found = 0
    for length in range(1, max_len + 1):
        for combo in itertools.product(charset, repeat=length):
            word = ''.join(combo)
            val = get_base22_value(word)
            if is_prime(val):
                update_cb.emit(f"{word} → {val} ✅")
                found += 1
    done_cb.emit(f"\n🧱 Total primes found: {found}")

class MinerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()

        self.label = QLabel("🔍 Prime Miner (Base-22)")
        self.length_label = QLabel("Max Length (characters):")
        self.length_input = QLineEdit("4")
        self.mine_button = QPushButton("Mine Primes")
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.length_label)
        self.layout.addWidget(self.length_input)
        self.layout.addWidget(self.mine_button)
        self.layout.addWidget(self.output_area)
        self.setLayout(self.layout)

        self.mine_button.clicked.connect(self.start_mining)

    def start_mining(self):
        self.output_area.clear()
        try:
            max_len = int(self.length_input.text())
        except ValueError:
            self.output_area.setText("❌ Invalid length input. Please enter a number.")
            return

        self.worker = Worker(run_prime_miner, max_len)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.update.connect(self.output_area.append)
        self.worker.done.connect(self.output_area.append)
        self.worker.done.connect(self.thread.quit)
        self.thread.start()