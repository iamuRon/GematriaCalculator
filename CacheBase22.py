from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHBoxLayout
)
from PyQt5.QtGui import QFont
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

        # Keyboard UI
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

        # Add spacebar
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

        value = get_linear_gematria(text)
        prime_status = "✅Prime!✅" if is_prime(value) else "Composite"
        result_display = f"{value} ({prime_status})"

        self.result_label.setText(f"Result: {result_display}")
        self.log_area.append(f"{text} → {value} ({prime_status})")
        self.entries.append((text, value, prime_status))

    def save_logs(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        txt_filename = f"gematria_log_{timestamp}.txt"
        csv_filename = f"gematria_log_{timestamp}.csv"

        try:
            with open(txt_filename, 'w', encoding='utf-8') as txt_file:
                for entry in self.entries:
                    txt_file.write(f"{entry[0]} → {entry[1]} ({entry[2]})\n")

            with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Input', 'Gematria Value', 'Prime?'])
                writer.writerows(self.entries)

            self.result_label.setText(f"✅ Logs saved as {txt_filename} + {csv_filename}")
        except Exception as e:
            self.result_label.setText(f"❌ Error saving: {e}")


class Base22Tab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        self.input_label = QLabel("Enter Hebrew text for Base-22 calculation:")
        self.input_field = QLineEdit()
        self.calculate_button = QPushButton("Calculate Base-22 Value")
        self.result_label = QLabel("Result: —")

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.result_label)

        self.setLayout(layout)

        self.calculate_button.clicked.connect(self.calculate_base22)

    def calculate_base22(self):
        text = self.input_field.text().strip()
        if not text:
            self.result_label.setText("Result: (Empty input)")
            return

        value = get_base22_value(text)
        prime_status = "✅Prime!✅" if is_prime(value) else "Composite"
        self.result_label.setText(f"Base-22: {value} ({prime_status})")


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

        self.mine_button.clicked.connect(self.mine_primes)

    def mine_primes(self):
        self.output_area.clear()
        try:
            max_len = int(self.length_input.text())
        except ValueError:
            self.output_area.setText("❌ Invalid length input. Please enter a number.")
            return

        charset = list(linear_map.keys())
        found = 0

        for length in range(1, max_len + 1):
            for combo in itertools.product(charset, repeat=length):
                word = ''.join(combo)
                val = get_base22_value(word)
                if is_prime(val):
                    self.output_area.append(f"{word} → {val} ✅")
                    found += 1

        self.output_area.append(f"\n🧱 Total primes found: {found}")


class InfoTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("📚 Aleph-Bet Linear Gematria Table (Aleph = 1 to Tav = 22)")
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

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧮 Hebrew Linear Gematria Tool")
        self.setGeometry(200, 200, 500, 500)

        self.gematria_tab = GematriaTab()
        self.base22_tab = Base22Tab()
        self.miner_tab = MinerTab()
        self.info_tab = InfoTab()
        self.keyboard_tab = KeyboardTab()

        self.addTab(self.gematria_tab, "Calculator")
        self.addTab(self.base22_tab, "Base-22 Calculator")
        self.addTab(self.miner_tab, "Prime Miner")
        self.addTab(self.info_tab, "Info")
        self.addTab(self.keyboard_tab, "Hebrew Keyboard")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())