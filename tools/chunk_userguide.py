import sys
import os
import shutil
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QPushButton, QLabel, QTextEdit, QFileDialog, 
                              QMessageBox, QFrame)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QClipboard

class MarkdownChunker(QMainWindow):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Markdown File Chunker")
       self.setGeometry(100, 100, 900, 700)
       
       self.longest_chunk = ""
       self.shortest_chunk = ""
       self.longest_length = 0
       self.shortest_length = float('inf')
       self.selected_file = None
       
       self.setup_ui()
   
   def setup_ui(self):
       central_widget = QWidget()
       self.setCentralWidget(central_widget)
       
       layout = QVBoxLayout(central_widget)
       layout.setContentsMargins(20, 20, 20, 20)
       layout.setSpacing(15)
       
       title_label = QLabel("Markdown File Chunker")
       title_font = QFont()
       title_font.setPointSize(16)
       title_font.setBold(True)
       title_label.setFont(title_font)
       title_label.setAlignment(Qt.AlignCenter)
       layout.addWidget(title_label)
       
       file_frame = QFrame()
       file_layout = QHBoxLayout(file_frame)
       
       self.file_label = QLabel("No file selected")
       self.file_label.setWordWrap(True)
       file_layout.addWidget(self.file_label, 1)
       
       select_button = QPushButton("Select Markdown File")
       select_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
       select_button.clicked.connect(self.select_file)
       file_layout.addWidget(select_button)
       
       layout.addWidget(file_frame)
       
       self.process_button = QPushButton("Process File")
       self.process_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 10px; font-size: 12px; }")
       self.process_button.setEnabled(False)
       self.process_button.clicked.connect(self.process_file)
       layout.addWidget(self.process_button)
       
       self.stats_label = QLabel("")
       stats_font = QFont()
       stats_font.setBold(True)
       self.stats_label.setFont(stats_font)
       layout.addWidget(self.stats_label)
       
       longest_label = QLabel("Longest Chunk:")
       longest_font = QFont()
       longest_font.setBold(True)
       longest_label.setFont(longest_font)
       layout.addWidget(longest_label)
       
       self.longest_text = QTextEdit()
       self.longest_text.setMaximumHeight(150)
       self.longest_text.setReadOnly(True)
       layout.addWidget(self.longest_text)
       
       shortest_label = QLabel("Shortest Chunk:")
       shortest_font = QFont()
       shortest_font.setBold(True)
       shortest_label.setFont(shortest_font)
       layout.addWidget(shortest_label)
       
       self.shortest_text = QTextEdit()
       self.shortest_text.setMaximumHeight(150)
       self.shortest_text.setReadOnly(True)
       layout.addWidget(self.shortest_text)
   
   def select_file(self):
       file_path, _ = QFileDialog.getOpenFileName(
           self,
           "Select Markdown File",
           "",
           "Markdown files (*.md);;Text files (*.txt);;All files (*.*)"
       )
       
       if file_path:
           self.selected_file = file_path
           self.file_label.setText(f"Selected: {os.path.basename(file_path)}")
           self.process_button.setEnabled(True)
   
   def create_output_directory(self):
       current_dir = os.getcwd()
       assets_dir = os.path.join(current_dir, "Assets")
       chunks_dir = os.path.join(assets_dir, "User_Guide_Chunks")
       
       if not os.path.exists(assets_dir):
           os.makedirs(assets_dir)
       
       if os.path.exists(chunks_dir):
           for filename in os.listdir(chunks_dir):
               file_path = os.path.join(chunks_dir, filename)
               try:
                   if os.path.isfile(file_path):
                       os.unlink(file_path)
               except Exception as e:
                   print(f"Error deleting {file_path}: {e}")
       else:
           os.makedirs(chunks_dir)
       
       return chunks_dir
   
   def extract_chunks(self, content):
       chunks = []
       
       lines = content.split('\n')
       current_chunk = []
       
       for line in lines:
           if line.strip().startswith('###'):
               if current_chunk:
                   chunk_text = '\n'.join(current_chunk).strip()
                   if chunk_text:
                       chunks.append(chunk_text)
               
               current_chunk = [line]
           elif current_chunk:
               if line.strip() or len(current_chunk) == 1:
                   current_chunk.append(line)
               else:
                   continue
       
       if current_chunk:
           chunk_text = '\n'.join(current_chunk).strip()
           if chunk_text:
               chunks.append(chunk_text)
       
       return chunks
   
   def save_chunks(self, chunks, output_dir):
       for i, chunk in enumerate(chunks, 1):
           filename = f"chunk_{i:03d}.txt"
           filepath = os.path.join(output_dir, filename)
           
           with open(filepath, 'w', encoding='utf-8') as f:
               f.write(chunk)
       
       return len(chunks)
   
   def analyze_chunks(self, chunks):
       if not chunks:
           return
       
       self.longest_chunk = chunks[0]
       self.shortest_chunk = chunks[0]
       self.longest_length = len(chunks[0])
       self.shortest_length = len(chunks[0])
       
       for chunk in chunks:
           chunk_length = len(chunk)
           
           if chunk_length > self.longest_length:
               self.longest_length = chunk_length
               self.longest_chunk = chunk
           
           if chunk_length < self.shortest_length:
               self.shortest_length = chunk_length
               self.shortest_chunk = chunk
   
   def create_master_questions(self, chunks):
       master_questions = []
       
       for chunk in chunks:
           lines = chunk.split('\n')
           first_line = lines[0].strip()
           if first_line.startswith('###'):
               question = first_line[3:].strip()
               master_questions.append(question)
       
       dictionary_str = "master_questions = [\n"
       for question in master_questions:
           dictionary_str += f'    "{question}",\n'
       dictionary_str += "]"
       
       clipboard = QApplication.clipboard()
       clipboard.setText(dictionary_str)
   
   def update_display(self, num_chunks):
       stats_text = f"Processing complete! Created {num_chunks} chunks.\n"
       stats_text += f"Longest chunk: {self.longest_length} characters\n"
       stats_text += f"Shortest chunk: {self.shortest_length} characters"
       self.stats_label.setText(stats_text)
       
       self.longest_text.setPlainText(self.longest_chunk)
       self.shortest_text.setPlainText(self.shortest_chunk)
   
   def process_file(self):
       if not self.selected_file:
           QMessageBox.critical(self, "Error", "Please select a file first.")
           return
       
       try:
           with open(self.selected_file, 'r', encoding='utf-8') as f:
               content = f.read()
           
           output_dir = self.create_output_directory()
           
           chunks = self.extract_chunks(content)
           
           if not chunks:
               QMessageBox.warning(self, "Warning", "No chunks found in the file.")
               return
           
           self.analyze_chunks(chunks)
           
           num_chunks = self.save_chunks(chunks, output_dir)
           
           self.create_master_questions(chunks)
           
           self.update_display(num_chunks)
           
           QMessageBox.information(self, "Success", 
                                 f"Successfully processed {num_chunks} chunks!\n"
                                 f"Files saved to: {output_dir}\n"
                                 f"Dictionary copied to clipboard!")
       
       except Exception as e:
           QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

def main():
   app = QApplication(sys.argv)
   window = MarkdownChunker()
   window.show()
   sys.exit(app.exec())

if __name__ == "__main__":
   main()