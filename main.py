import sys
import os
import time
import ctypes
import json
import msgpack
from multiprocessing import Process, Queue
from elevenlabs import generate, play, stream, Voice, VoiceSettings, set_api_key
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

user32 = ctypes.WinDLL('user32', use_last_error=True)


def keybd_event(bVk, bScan, dwFlags, dwExtraInfo):
    user32.keybd_event(bVk, bScan, dwFlags, dwExtraInfo)


# Constants for the Play/Pause key
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_EXTENDEDKEY = 0x1
KEYEVENTF_KEYUP = 0x2


def simulate_media_play_pause():
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0,
                KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

def tts_task(texts, api_key, queue):
    try:
        set_api_key(api_key)
        for text in texts:
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id='CXFz1nBWCcj72XaSOBwT',
                    settings=VoiceSettings(
                        stability=0.75, similarity_boost=0.6, style=0.5, use_speaker_boost=True)
                ),
                model="eleven_turbo_v2",
                stream=True
            )
            stream(audio)

    except Exception as e:
        queue.put(str(e))
    finally:
        queue.put("done")


class GPTWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, message, conversation_history, prompt_type="default"):
        super().__init__()
        self.message = message
        self.conversation_history = conversation_history
        self.prompt_type = prompt_type
        self.system_prompts = self.load_system_prompts()

    @staticmethod
    def load_system_prompts():
        prompts = {}
        prompt_dir = 'prompts/'  # Update with the correct path
        for filename in os.listdir(prompt_dir):
            if filename.endswith('.txt'):
                prompt_type = filename.rsplit('.', 1)[0]  # Get the file name without the extension
                with open(os.path.join(prompt_dir, filename), 'r') as file:
                    prompts[prompt_type] = file.read().strip()
        return prompts

    def run(self):
        try:
            system_prompt = self.system_prompts.get(self.prompt_type, "default")
            messages = [
                {"role": "system", "content": system_prompt + """(With each response add an expression from 'Normal, Surprised, Love, Happy, Confused, Angry' 
                 to the start of the message in square brackets, use the often and don't use the same one more than twice in a row.)"""}
            ]
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


class TTSWorker:
    def __init__(self, texts):
        self.texts = texts
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.queue = Queue()

    def start(self):
        self.process = Process(target=tts_task, args=(
            self.texts, self.api_key, self.queue))  # Pass the list of texts here
        self.process.start()

    def is_running(self):
        return self.process.is_alive()

    def terminate(self):
        if self.is_running():
            self.process.terminate()


class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, virtual_assistant, parent=None):
        super().__init__(pixmap, parent)
        self.virtual_assistant = virtual_assistant

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.virtual_assistant.cycle_outfit()
        elif event.button() == Qt.RightButton:
            self.virtual_assistant.reverse_cycle_outfit()

class VirtualAssistant(QMainWindow):
    def __init__(self, blink_speed=35, blink_timer=4000):
        super().__init__()
        
        self.current_prompt_type = "default"
        self.current_outfit = 'default'
        self.current_expression = 'normal'
        self.isBlinking = False
        self.blinking_index = 0

        self.current = volume.GetMasterVolumeLevel()

        self.conversation_history = []

        # Define outfits and expressions
        self.outfits = ['default', 'cat', 'devil', 'mini', 'victorian',
                        'chinese', 'yukata', 'steampunk', 'gown', 'bikini', 'cyberpunk']
        self.expressions = ['normal', 'surprised', 'love', 'happy',
                            'confused', 'angry']

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
                        int(QPixmap(
                            f'sprites/{outfit}/{expression}Blink{i}.png').width()),
                        int(QPixmap(
                            f'sprites/{outfit}/{expression}Blink{i}.png').height()),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    for i in range(1, 4)]

        # Timer for blinking animation
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.blink)
        self.blink_timer.start(blink_timer)

        # Timer for blinking animation frames
        self.blink_frame_timer = QTimer(self)
        self.blink_frame_timer.timeout.connect(self.blink_frame)
        self.blink_frame_speed = blink_speed

        # Set up the rest of the window
        self.setFixedSize(400, 440)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.view = QGraphicsView(self)
        self.view.setGeometry(0, 0, 400, 450)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Initialize the sprite item with the current outfit and expression
        self.init_sprite_item()

        # Load the config and conversation history
        self.config_file = 'config.json'
        self.load_config()
        self.load_conversation_history()

        # Position the window at the bottom right of the screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = screen_geometry.width() - self.width()
        y = screen_geometry.height() - self.height()
        self.move(x, y)

        # Add a chat box
        self.chat_box = QLineEdit(self)
        self.chat_box.setGeometry(10, 400, 380, 25)
        self.chat_box.setStyleSheet(
            "background-color: white; color: black; border: 2px solid black; border-radius: 5px;")
        self.chat_box.setPlaceholderText("Use the 'gpt:' prefix for ChatGPT.")
        self.chat_box.setFocus()
        self.chat_box.returnPressed.connect(self.process_command)

    def closeEvent(self, event):
        # Terminate the TTSWorker process if it's running
        if hasattr(self, 'tts_worker'):
            try:
                if self.tts_worker.is_running():
                    self.tts_worker.terminate()
                self.tts_worker.process.join()  # Wait for the process to terminate
            except Exception as e:
                print(f"Error while terminating TTSWorker: {e}")

        # Terminate the GPTWorker thread if it's running
        if hasattr(self, 'gpt_worker') and self.gpt_worker.isRunning():
            self.gpt_worker.terminate()
            self.gpt_worker.wait()

        event.accept()

    def save_conversation_history(self):
        with open('conversation_history.msgpack', 'wb') as f:
            msgpack.dump(self.conversation_history, f)

    def load_conversation_history(self):
        if os.path.exists('conversation_history.msgpack'):
            with open('conversation_history.msgpack', 'rb') as f:
                self.conversation_history = msgpack.load(f)

    def init_sprite_item(self):
        initial_pixmap = self.sprites[self.current_outfit][self.current_expression]
        self.sprite_item = ClickablePixmapItem(initial_pixmap, self)
        self.scene.addItem(self.sprite_item)

    def save_config(self):
        config = {
            'outfit': self.current_outfit,
            'prompt_type': self.current_prompt_type
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                self.current_outfit = config.get('outfit', 'default')
                self.current_prompt_type = config.get('prompt_type', 'default')
                self.update_sprite()  # Update the sprite with the loaded outfit
        else:
            self.current_outfit = 'default'
            self.current_prompt_type = 'default'

    def cycle_outfit(self):
        current_index = self.outfits.index(self.current_outfit)
        new_index = (current_index + 1) % len(self.outfits)
        self.current_outfit = self.outfits[new_index]
        self.update_sprite()
        self.save_config()

    def reverse_cycle_outfit(self):
        current_index = self.outfits.index(self.current_outfit)
        new_index = (current_index - 1) % len(self.outfits)
        self.current_outfit = self.outfits[new_index]
        self.update_sprite()
        self.save_config()

    def change_outfit(self, new_outfit):
        if new_outfit in self.outfits:
            self.current_outfit = new_outfit
            self.update_sprite()
            self.save_config()

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

    def blink_frame(self, manual_control=False):
        if self.isBlinking or manual_control:
            blink_sequence = self.blinking_sprites[self.current_outfit][self.current_expression]

            if self.blinking_index < len(blink_sequence):
                self.sprite_item.setPixmap(blink_sequence[self.blinking_index])
                self.blinking_index += 1
            else:
                self.blinking_index = 0
                self.isBlinking = False
                self.blink_frame_timer.stop() if not manual_control else None
                self.sprite_item.setPixmap(self.sprites[self.current_outfit][self.current_expression])

    def test_expressions_and_blink(self):
        for expression in self.expressions:
            self.change_expression(expression)
            QApplication.processEvents()
            time.sleep(1)

            # Manually trigger and control the blink animation
            self.isBlinking = True
            for frame_index in range(4):  # Iterate through frames 0 to 4
                self.blinking_index = frame_index
                self.blink_frame(manual_control=True)
                QApplication.processEvents()
                time.sleep(0.25)  # Adjust timing as needed

            # Reset to the normal state
            self.isBlinking = False
            self.blinking_index = 0
            self.sprite_item.setPixmap(self.sprites[self.current_outfit][self.current_expression])
            QApplication.processEvents()

    def process_command(self):
        command = self.chat_box.text().strip().lower()

        # Define a prefix for chatgpt
        prefix = "gpt:"

        if command == "quit" or command == "close" or command == "exit" or command == "q":
            self.close()
            return
        elif command == "volume up":
            volume.SetMasterVolumeLevelScalar(
                volume.GetMasterVolumeLevelScalar() + 0.1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chat_box.clear()
            return
        elif command == "volume down":
            volume.SetMasterVolumeLevelScalar(
                volume.GetMasterVolumeLevelScalar() - 0.1, None)
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
                    self.chat_box.setText(
                        "Invalid volume percentage. Please use a value between 0 and 100.")
            except ValueError:
                self.chat_box.setText(
                    "Invalid volume percentage. Please use a numeric value.")
            return
        elif command == "hide":
            # Disable always on top
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            self.setVisible(True)
            self.showMinimized()
            self.chat_box.clear()
            return
        elif command == "show":
            self.show()
            self.setWindowFlag(Qt.WindowStaysOnTopHint,
                               True)  # Enable always on top
            self.setVisible(True)
            self.chat_box.clear()
            return
        elif command == "pause" or command == "play" or command == "p":
            simulate_media_play_pause()
            self.chat_box.clear()
            return
        elif command.lower().startswith("switch: "):
            new_prompt_type = command[len("switch: "):].strip()
            self.current_prompt_type = new_prompt_type
            self.save_config()
            self.chat_box.clear()
            return
        elif command == "test":
            self.test_expressions_and_blink()
            self.chat_box.clear()
            return

        # Check if the command starts with the prefix
        if not command.lower().startswith(prefix.lower()):
            self.chat_box.clear()
            return

        # Remove the prefix from the command
        command = command[len(prefix):].strip().lower()

        # Add user command to history
        self.conversation_history.append({"role": "user", "content": command})

        # Save conversation history
        self.save_conversation_history()

        max_history_length = 100
        if len(self.conversation_history) > max_history_length:
            self.conversation_history = self.conversation_history[-max_history_length:]

        self.gpt_worker = GPTWorker(command, self.conversation_history, prompt_type=self.current_prompt_type)
        self.gpt_worker.finished.connect(self.handle_gpt_response)
        self.gpt_worker.start()

        self.chat_box.clear()

    def handle_gpt_response(self, gpt_response):
        self.conversation_history.append(
            {"role": "assistant", "content": gpt_response})

        # Save conversation history
        self.save_conversation_history()

        if gpt_response.startswith('[') and ']' in gpt_response:
            end_bracket_index = gpt_response.find(']')
            # Convert expression to lowercase
            expression = gpt_response[1:end_bracket_index].strip().lower()
            message = gpt_response[end_bracket_index + 1:].strip()

            if expression in self.expressions:
                self.change_expression(expression)
            else:
                message = gpt_response  # Use the original message if expression is not valid
        else:
            message = gpt_response  # Use the original message if no expression is found

        messages = [message]  # Wrap the response in a list

        # Start the TTS worker
        print(message)
        self.tts_worker = TTSWorker(messages)
        self.tts_worker.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    blink_speed = 25
    blink_timer = 4000
    assistant = VirtualAssistant(blink_speed, blink_timer)
    assistant.show()
    sys.exit(app.exec_())
