from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QTextEdit,
    QTabWidget, QTableWidget, QTableWidgetItem, QHBoxLayout
)
import sys
import datetime
import csv

def get_linear_gematria(text):
    linear_map = {
        'א': 1,  'ב': 2,  'ג': 3,  'ד': 4,  'ה': 5,  'ו': 6,
        'ז': 7,  'ח': 8,  'ט': 9,  'י': 10, 'כ': 11, 'ך': 11,
        'ל': 12, 'מ': 13, 'ם': 13, 'נ': 14, 'ן': 14, 'ס': 15,
        'ע': 16, 'פ': 17, 'ף': 17, 'צ': 18, 'ץ': 18, 'ק': 19,
        'ר': 20, 'ש': 21, 'ת': 22
    }

    total = 0
    for letter in text:
        if letter in linear_map:
            total += linear_map[letter]
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

        layout = QVBoxLayout()
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_field)
        layout.addWidget(self.calculate_button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.log_area)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

        self.calculate_button.clicked.connect(self.calculate_gematria)
        self.save_button.clicked.connect(self.save_logs)

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

class InfoTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        label = QLabel("📚 Aleph-Bet Linear Gematria Table (Aleph = 1 to Tav = 22)")
        layout.addWidget(label)

        table = QTableWidget()
        linear_map = [
            ('א', 1),  ('ב', 2),  ('ג', 3),  ('ד', 4),  ('ה', 5),  ('ו', 6),
            ('ז', 7),  ('ח', 8),  ('ט', 9),  ('י', 10), ('כ', 11), ('ך', 11),
            ('ל', 12), ('מ', 13), ('ם', 13), ('נ', 14), ('ן', 14), ('ס', 15),
            ('ע', 16), ('פ', 17), ('ף', 17), ('צ', 18), ('ץ', 18), ('ק', 19),
            ('ר', 20), ('ש', 21), ('ת', 22)
        ]

        table.setRowCount(len(linear_map))
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Letter", "Value"])

        for i, (letter, value) in enumerate(linear_map):
            table.setItem(i, 0, QTableWidgetItem(letter))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

        table.resizeColumnsToContents()
        layout.addWidget(table)
        self.setLayout(layout)

class MainWindow(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧮 Hebrew Linear Gematria Tool")
        self.setGeometry(200, 200, 500, 500)

        self.gematria_tab = GematriaTab()
        self.info_tab = InfoTab()

        self.addTab(self.gematria_tab, "Calculator")
        self.addTab(self.info_tab, "Info")

# Run the App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
