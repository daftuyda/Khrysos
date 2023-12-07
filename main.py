import sys
import os
import pyttsx3
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from dotenv import load_dotenv
from openai import OpenAI
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

load_dotenv()

openai_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_key)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Worker class for handling GPT communication
class GPTWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, message, conversation_history):
        super().__init__()
        self.message = message
        self.conversation_history = conversation_history

    def run(self):
        try:
            messages=[{"role": "system", "content": "Your name is Yuki, you are my personal assistant. Use casual language when responding. Keep responses short and concise. (With each response add an expression from 'Normal, Surprised, Love' to the start of the message in square brackets and change them often or at random.)"}]
            messages += self.conversation_history
            messages.append({"role": "user", "content": self.message})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            assistant_message = response.choices[0].message.content.strip()
            self.finished.emit(assistant_message)
        except Exception as e:
            self.finished.emit(str(e))

class TTSWorker(QThread):
    def __init__(self, tts_engine, text):
        super().__init__()
        self.tts_engine = tts_engine
        self.text = text

    def run(self):
        try:
            print(f"Speaking: {self.text}")  # Debug print
            self.tts_engine.say(self.text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")  # Log any exceptions

class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, virtual_pet, parent=None):
        super().__init__(pixmap, parent)
        self.virtual_pet = virtual_pet

    def mousePressEvent(self, event):
        self.virtual_pet.cycle_outfit()

# Main VirtualPet class
class VirtualPet(QMainWindow):
    def __init__(self, blink_speed=35, blink_timer=4000, speech_rate=150):
        super().__init__()

        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id)  # Set the desired voice
        self.tts_engine.setProperty('rate', speech_rate)  # Set speech rate (speed)

        self.current_outfit = 'default'
        self.current_expression = 'normal'
        self.isBlinking = False
        self.blinking_index = 0

        self.current = volume.GetMasterVolumeLevel()

        self.conversation_history = []

        # Define possible outfits and expressions
        self.outfits = ['default', 'cat', 'devil']
        self.expressions = ['normal', 'surprised', 'love']

        # Load sprites and blinking animation sprites
        self.sprites = {}
        self.blinking_sprites = {}
        for outfit in self.outfits:
            self.sprites[outfit] = {}
            self.blinking_sprites[outfit] = {}
            for expression in self.expressions:
                # Load regular sprites
                sprite_path = f'sprites/{outfit}/{expression}.png'
                self.sprites[outfit][expression] = QPixmap(sprite_path).scaled(
                    int(QPixmap(sprite_path).width()),
                    int(QPixmap(sprite_path).height()),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Load blinking animation sprites
                self.blinking_sprites[outfit][expression] = [
                    QPixmap(f'sprites/{outfit}/{expression}Blink{i}.png').scaled(
                        int(QPixmap(f'sprites/{outfit}/{expression}Blink{i}.png').width()),
                        int(QPixmap(f'sprites/{outfit}/{expression}Blink{i}.png').height()),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    for i in range(1, 4)]  # Assuming 3 frames for blinking animation

        # Timer for blinking animation
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink)
        self.blink_timer.start(blink_timer)  # Interval for starting blink

        # Timer for blinking animation frames
        self.blink_frame_timer = QTimer(self)
        self.blink_frame_timer.timeout.connect(self.blink_frame)
        self.blink_frame_speed = blink_speed  # Milliseconds per frame

        # Set up the rest of the window
        self.setFixedSize(400, 440)  # Adjust size as needed
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, 400, 450)  # Adjust size as needed
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Initialize the sprite item with the current outfit and expression
        self.init_sprite_item()

        # Position the window at the bottom right of the screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = screen_geometry.width() - self.width()
        y = screen_geometry.height() - self.height()
        self.move(x, y)

        # Add a chat box
        self.chat_box = QLineEdit(self)
        self.chat_box.setGeometry(10, 400, 380, 25)
        self.chat_box.setStyleSheet("background-color: white; color: black; border: 2px solid black; border-radius: 5px;")
        self.chat_box.returnPressed.connect(self.process_command)  # Connect the returnPressed signal

    def init_sprite_item(self):
        initial_pixmap = self.sprites[self.current_outfit][self.current_expression]
        self.sprite_item = ClickablePixmapItem(initial_pixmap, self)
        self.scene.addItem(self.sprite_item)

    def cycle_outfit(self):
        # Cycle through the outfits
        current_index = self.outfits.index(self.current_outfit)
        new_index = (current_index + 1) % len(self.outfits)
        self.current_outfit = self.outfits[new_index]
        self.update_sprite()

    def change_outfit(self, new_outfit):
        if new_outfit in self.outfits:
            self.current_outfit = new_outfit
            self.update_sprite()

    def change_expression(self, new_expression):
        if new_expression in self.expressions:
            self.current_expression = new_expression
            self.update_sprite()

    def update_sprite(self):
        new_pixmap = self.sprites[self.current_outfit][self.current_expression]
        self.sprite_item.setPixmap(new_pixmap)

    def blink(self):
        self.isBlinking = True
        self.blinking_index = 0
        self.blink_frame_timer.start(self.blink_frame_speed)

    def blink_frame(self):
        if self.isBlinking:
            blink_sequence = self.blinking_sprites[self.current_outfit][self.current_expression]

            # Ensure the index is within the range of blink_sequence
            if self.blinking_index < len(blink_sequence):
                self.sprite_item.setPixmap(blink_sequence[self.blinking_index])
                self.blinking_index += 1
            else:
                # Reset blinking state
                self.blinking_index = 0
                self.isBlinking = False
                self.blink_frame_timer.stop()
                self.sprite_item.setPixmap(self.sprites[self.current_outfit][self.current_expression])

    def process_command(self):
        command = self.chat_box.text().strip().lower()

        # Define a prefix for chatgpt
        prefix = "gpt:"

        if command == "quit" or command== "close" or command == "exit" or command == "q":
            self.close()
            return
        elif command == "volume up":
            volume.SetMasterVolumeLevelScalar(volume.GetMasterVolumeLevelScalar() + 0.1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command == "volume down":
            volume.SetMasterVolumeLevelScalar(volume.GetMasterVolumeLevelScalar() - 0.1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command == "volume mute" or command == "volume 0":
            volume.SetMasterVolumeLevelScalar(0, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command == "volume max" or command == "volume 1":
            volume.SetMasterVolumeLevelScalar(1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command == "volume mid":
            volume.SetMasterVolumeLevelScalar(0.5, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command.startswith("volume "):
            try:
                percent = float(command.split("volume ")[1])
                if 0 <= percent <= 100:
                    # Convert percentage to a value between 0 and 1
                    volume_level = percent / 100.0
                    volume.SetMasterVolumeLevelScalar(volume_level, None)
                    self.current = volume.GetMasterVolumeLevel()
                    self.chat_box.clear()
                else:
                    self.chat_box.setText("Invalid volume percentage. Please use a value between 0 and 100.")
            except ValueError:
                self.chat_box.setText("Invalid volume percentage. Please use a numeric value.")
            return
        elif command == "hide":
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)  # Disable always on top
            self.setVisible(True)
            self.chat_box.clear()
            return
        elif command == "show":
            self.show()
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)  # Enable always on top
            self.setVisible(True)
            self.chat_box.clear()
            return

        # Check if the command starts with the prefix
        if not command.lower().startswith(prefix.lower()):
            self.chat_box.clear()
            self.chat_box.setPlaceholderText("Use the 'gpt:' prefix for ChatGPT.")
            return

        # Remove the prefix from the command
        command = command[len(prefix):].strip().lower()

        # Add user command to history
        self.conversation_history.append({"role": "user", "content": command})

        max_history_length = 25  # Keep last 25 exchanges
        if len(self.conversation_history) > max_history_length:
            self.conversation_history = self.conversation_history[-max_history_length:]

        self.gpt_worker = GPTWorker(command, self.conversation_history)
        self.gpt_worker.finished.connect(self.handle_gpt_response)
        self.gpt_worker.start()

        self.chat_box.clear()

    def handle_gpt_response(self, gpt_response):
        #print(f"Original Response: {gpt_response}")
        self.conversation_history.append({"role": "assistant", "content": gpt_response})

        if gpt_response.startswith('[') and ']' in gpt_response:
            end_bracket_index = gpt_response.find(']')
            expression = gpt_response[1:end_bracket_index].strip().lower()  # Convert expression to lowercase
            message = gpt_response[end_bracket_index + 1:].strip()

            if expression in self.expressions:
                self.change_expression(expression)
            else:
                message = gpt_response  # Use the original message if expression is not valid
        else:
            message = gpt_response  # Use the original message if no expression is found

        # Start the TTS worker
        self.tts_worker = TTSWorker(self.tts_engine, message)
        self.tts_worker.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    blink_speed = 25
    blink_timer = 4000
    speech_rate = 225
    pet = VirtualPet(blink_speed, blink_timer, speech_rate)
    pet.show()
    sys.exit(app.exec_())
