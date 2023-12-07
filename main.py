import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt

class VirtualPet(QMainWindow):
    def __init__(self, scale_factor=1.0):
        super().__init__()

        # Define states
        self.states = ['normal', 'surprised', 'cat', 'devil']
        
        # Load and scale sprites for each state
        self.sprites = {state: QPixmap(f'{state}.png').scaled(
                        int(QPixmap(f'{state}.png').width() * scale_factor),
                        int(QPixmap(f'{state}.png').height() * scale_factor),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        for state in self.states}

        # Define and load blinking sprites for each state
        self.blinking_sprites = {}
        for state in self.states:
            self.blinking_sprites[state] = [QPixmap(f'{state}Blink{i}.png').scaled(
                                            int(QPixmap(f'{state}Blink{i}.png').width() * scale_factor),
                                            int(QPixmap(f'{state}Blink{i}.png').height() * scale_factor),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
                                            for i in range(1, 4)]  # Assuming 3 blinking frames per state

        self.current_state = 'normal'
        self.isBlinking = False
        self.blinking_index = 0

        # Set up the rest of the window
        self.setFixedSize(self.sprites['normal'].width() + 300, self.sprites['normal'].height())
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, self.sprites['normal'].width(), self.sprites['normal'].height())
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.sprite_item = QGraphicsPixmapItem(self.sprites[self.current_state])
        self.scene.addItem(self.sprite_item)
        self.timer = QTimer(self)
        self.timer.start(100)
        screen_geometry = QApplication.desktop().screenGeometry()
        x = screen_geometry.width() - self.width()
        y = screen_geometry.height() - self.height()
        self.move(x, y)

        # Timer for initiating blinking
        self.blink_init_timer = QTimer(self)
        self.blink_init_timer.timeout.connect(self.start_blinking)
        self.blink_init_timer.start(5000)  # Interval between blinks

        # Timer for blinking animation frames
        self.blink_frame_timer = QTimer(self)
        self.blink_frame_timer.timeout.connect(self.blink_frame)
        self.blink_frame_speed = 100  # Milliseconds per frame, adjust for faster animation

        # Add a chat box
        self.chat_box = QLineEdit(self)
        self.chat_box.setGeometry(self.sprites['normal'].width() + 10, 10, 280, 30)

        # Add a submit button
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setGeometry(self.sprites['normal'].width() + 10, 50, 280, 30)
        self.submit_button.clicked.connect(self.process_command)

    def change_state(self, new_state):
        if new_state in self.sprites:
            self.current_state = new_state
            self.sprite_item.setPixmap(self.sprites[new_state])
            self.isBlinking = False  # Reset blinking when state changes

    def start_blinking(self):
        self.isBlinking = True
        self.blinking_index = 0
        self.blink_frame_timer.start(self.blink_frame_speed)

    def blink_frame(self):
        if self.isBlinking:
            blink_sequence = self.blinking_sprites[self.current_state]
            self.sprite_item.setPixmap(blink_sequence[self.blinking_index])
            self.blinking_index = (self.blinking_index + 1) % len(blink_sequence)

            if self.blinking_index == 0:
                self.isBlinking = False
                self.blink_frame_timer.stop()
                self.sprite_item.setPixmap(self.sprites[self.current_state])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.current_state != 'normal':
                self.change_state('normal')
            else:
                self.change_state('surprised')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q and event.modifiers() == Qt.ShiftModifier:
            self.close()

    def process_command(self):
        command = self.chat_box.text().strip().lower()
        if command in self.sprites:
            self.change_state(command)
        self.chat_box.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    scale_factor = 1.0  
    pet = VirtualPet(scale_factor)
    pet.show()
    sys.exit(app.exec_())
