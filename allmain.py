import sys
import io
import os
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QDialog, QTextEdit,
    QLabel, QFileDialog, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# Importing script modules
import alldecition as abc1
import allann as abc3
import allrandom as abc4


class OutputWindow(QDialog):
    def __init__(self, title, output, image_paths=None):
        super().__init__()
        self.setWindowTitle(title)
        self.init_ui(output, image_paths)

    def init_ui(self, output, image_paths):
        layout = QVBoxLayout()

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(output)
        layout.addWidget(self.text_edit)

        if image_paths:
            for path in image_paths:
                self.image_label = QLabel(self)
                pixmap = QPixmap(path)
                self.image_label.setPixmap(pixmap)
                layout.addWidget(self.image_label)

        self.setLayout(layout)

class FileUploadWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):

        layout = QVBoxLayout()

        title_label = QLabel('Please upload your database as CSV file!')
        title_label.setStyleSheet("color: #4682B4; font-size: 14px;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)
        self.setWindowTitle('Upload CSV File')
    
        self.setStyleSheet("background-color: #B0E0E6")
        self.upload_button = QPushButton('Upload Database', self)
        
        self.upload_button.setStyleSheet("color: white ; background-color: #4682B4")
        self.upload_button.clicked.connect(self.upload_csv)
        layout.addWidget(self.upload_button)

        self.setLayout(layout)
        self.setFixedSize(300, 200)

    def upload_csv(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            self.main_window.csv_file_path = file_path
            self.main_window.show()
            self.close()

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.csv_file_path = ""
        self.init_ui()
        self.current_output_window = None  # Attribute to keep track of the current window

    def init_ui(self):
        self.setWindowTitle('Script Runner')
        self.setStyleSheet("background-color: #4682B4;")

        layout = QVBoxLayout()

        # Title label
        title_label = QLabel('Choose the method:')
        title_label.setStyleSheet("color: white; font-size: 24px;")
        layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Spacer to push buttons higher
        spacer = QSpacerItem(100, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)  # Set spacing between buttons

        # Button 1
        self.button1 = QPushButton('Decision Tree')
        self.button1.setStyleSheet("color : #4682B4 ; font-size: 18px ;background-color: #B0E0E6;")
        self.button1.setFixedSize(150, 75)
        self.button1.clicked.connect(self.run_code_1)
        button_layout.addWidget(self.button1)

        

        # Button 3
        self.button3 = QPushButton('Neural Network')
        self.button3.setStyleSheet("color : #4682B4 ; font-size: 18px ;background-color: #B0E0E6;")
        self.button3.setFixedSize(150, 75)
        self.button3.clicked.connect(self.run_code_3)
        button_layout.addWidget(self.button3)

        # Button 4
        self.button4 = QPushButton('Random Forest')
        self.button4.setStyleSheet("color : #4682B4 ; font-size: 18px ;background-color: #B0E0E6;")
        self.button4.setFixedSize(150, 75)
        self.button4.clicked.connect(self.run_code_4)
        button_layout.addWidget(self.button4)

        

        layout.addLayout(button_layout)
        self.setLayout(layout)
        self.setFixedSize(830, 300)  # Set window size

    def run_code_1(self):
        self.run_script(abc1, "Output of Decision Tree")

   

    def run_code_3(self):
        self.run_script(abc3, "Output of Neural Network")

    def run_code_4(self):
        self.run_script(abc4, "Output of Random Forest")

    

    def run_script(self, script_module, title):
        # Close the current output window if it exists
        if self.current_output_window:
            self.current_output_window.close()

        # Redirect stdout
        sys.stdout = io.StringIO()
        image_paths = None
        try:
            result = script_module.runscript(self.csv_file_path)
            if isinstance(result, tuple):
                if isinstance(result[0], list):
                    output = "Accuracies:\n" + "\n".join([f"k={i+1}: {acc:.2f}" for i, acc in enumerate(result[0])]) + "\n"
                else:
                    output = f"Accuracy: {result[0]:.2f}\n"
                image_paths = result[1:]
            else:
                output = sys.stdout.getvalue()
        finally:
            sys.stdout = sys.__stdout__  # Reset redirect

        # Show the new output window
        self.show_output_window(title, output, image_paths)

    def show_output_window(self, title, output, image_paths=None):
        self.current_output_window = OutputWindow(title, output, image_paths)
        self.current_output_window.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = MyWindow()
    file_upload_window = FileUploadWindow(main_window)
    file_upload_window.show()

    sys.exit(app.exec_())
