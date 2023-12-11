import sys
import os
import time
import ctypes
import json
import threading
import sqlite3
import requests
import keyboard
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from multiprocessing import Process, Queue
from elevenlabs import generate, stream, Voice, VoiceSettings, set_api_key
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from dotenv import load_dotenv
from openai import OpenAI
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit, QLabel
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal

load_dotenv()

openAiKey = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openAiKey)

clientId = os.getenv('SpotifyClientId')
secretId = os.getenv('SpotifySecretId')
redirectUri = os.getenv('SpotifyRedirectUri')
haUrl = os.getenv('haUrl')
haToken = os.getenv('haToken')

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

user32 = ctypes.WinDLL('user32', use_last_error=True)

scope = "user-read-private user-read-playback-state user-modify-playback-state"


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=clientId,
                                               client_secret=secretId,
                                               redirect_uri=redirectUri,
                                               scope=scope))


def search_and_play(song_name):
    results = sp.search(q=song_name, limit=1)
    if results['tracks']['items']:
        song_uri = results['tracks']['items'][0]['uri']
        sp.start_playback(uris=[song_uri])
        return f"Playing: {results['tracks']['items'][0]['name']}"
    else:
        return "Song not found."


def keybd_event(bVk, bScan, dwFlags, dwExtraInfo):
    user32.keybd_event(bVk, bScan, dwFlags, dwExtraInfo)


# Constants for the Play/Pause key
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_EXTENDEDKEY = 0x1
KEYEVENTF_KEYUP = 0x2


def simulatePlayPause():
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0,
                KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def ttsTask(texts, apiKey, queue):
    try:
        set_api_key(apiKey)
        for text in texts:
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id='EqudVpb9UKv174PwNcST',
                    settings=VoiceSettings(
                        stability=0.35, similarity_boost=0.4, style=0.0, use_speaker_boost=True)
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

    def __init__(self, message, conversationHistory, promptType="default"):
        super().__init__()
        self.message = message
        self.conversationHistory = conversationHistory
        self.promptType = promptType
        self.systemPrompts = self.loadSystemPrompts()

    @staticmethod
    def loadSystemPrompts():
        prompts = {}
        promptDir = 'prompts/'  # Update with the correct path
        for filename in os.listdir(promptDir):
            if filename.endswith('.txt'):
                # Get the file name without the extension
                promptType = filename.rsplit('.', 1)[0]
                with open(os.path.join(promptDir, filename), 'r') as file:
                    prompts[promptType] = file.read().strip()
        return prompts

    def run(self):
        try:
            systemPrompt = self.systemPrompts.get(
                self.promptType, "default")
            messages = [
                {"role": "system", "content": systemPrompt + """(With each response add an expression from 'Normal, Surprised, Love, Happy, Confused, Angry' 
                 to the start of the message in square brackets, use the often and don't use the same one more than twice in a row.) Keep messages to a 110 character limit if possible."""}
            ]
            messages += self.conversationHistory
            messages.append({"role": "user", "content": self.message})

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )

            assistantMessage = response.choices[0].message.content.strip()
            self.finished.emit(assistantMessage)
        except Exception as e:
            self.finished.emit(str(e))


class TTSWorker:
    def __init__(self, texts):
        self.texts = texts
        self.apiKey = os.getenv('ELEVENLABS_API_KEY')
        self.queue = Queue()

    def start(self):
        self.process = Process(target=ttsTask, args=(
            self.texts, self.apiKey, self.queue))  # Pass the list of texts here
        self.process.start()

    def isRunning(self):
        return self.process.is_alive()

    def terminate(self):
        if self.isRunning():
            self.process.terminate()


class ClickablePixmapItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, virtualAssistant, parent=None):
        super().__init__(pixmap, parent)
        self.virtualAssistant = virtualAssistant

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.virtualAssistant.cycleOutfit()
        elif event.button() == Qt.RightButton:
            self.virtualAssistant.reverseCycleOutfit()


class VirtualAssistant(QMainWindow):

    displayDelayedResponse = pyqtSignal(str)

    def __init__(self, blinkSpeed=35, blinkTimer=4000, delayDuration=5, bubbleTimerDuration=10000):
        super().__init__()

        self.dbConnection = sqlite3.connect('conversationHistory.db')
        self.dbCursor = self.dbConnection.cursor()

        self.currentPromptType = "default"
        self.currentOutfit = 'default'
        self.currentExpression = 'normal'
        self.isBlinking = False
        self.blinkingIndex = 0
        self.noTtsMode = False

        self.current = volume.GetMasterVolumeLevel()

        self.conversationHistory = []

        # Define outfits and expressions
        self.outfits = ['default', 'cat', 'devil', 'mini', 'victorian',
                        'chinese', 'yukata', 'steampunk', 'gown', 'bikini', 'cyberpunk']
        self.expressions = ['normal', 'surprised', 'love', 'happy',
                            'confused', 'angry']

        # Load sprites and blinking animation sprites
        self.sprites = {}
        self.blinkingSprites = {}
        for outfit in self.outfits:
            self.sprites[outfit] = {}
            self.blinkingSprites[outfit] = {}
            for expression in self.expressions:
                # Load regular sprites
                spritePath = f'sprites/{outfit}/{expression}.png'
                self.sprites[outfit][expression] = QPixmap(spritePath).scaled(
                    int(QPixmap(spritePath).width()),
                    int(QPixmap(spritePath).height()),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # Load blinking animation sprites
                self.blinkingSprites[outfit][expression] = [
                    QPixmap(f'sprites/{outfit}/{expression}Blink{i}.png').scaled(
                        int(QPixmap(
                            f'sprites/{outfit}/{expression}Blink{i}.png').width()),
                        int(QPixmap(
                            f'sprites/{outfit}/{expression}Blink{i}.png').height()),
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    for i in range(1, 4)]

        # Load the local font
        fontId = QFontDatabase.addApplicationFont("font/dogicapixel.ttf")
        if fontId == -1:
            print("Failed to load font")
        else:
            fontFamily = QFontDatabase.applicationFontFamilies(fontId)[0]
            customFont = QFont(fontFamily)
            customFont.setPointSize(6)
            customFont.setLetterSpacing(QFont.PercentageSpacing, 90)

        # Timer for blinking animation
        self.blinkTimer = QTimer(self)
        self.blinkTimer.timeout.connect(self.blink)
        self.blinkTimer.start(blinkTimer)

        # Timer for blinking animation frames
        self.blinkFrameTimer = QTimer(self)
        self.blinkFrameTimer.timeout.connect(self.blinkFrame)
        self.blinkFrameSpeed = blinkSpeed

        # Set the delay and bubble timer durations
        self.delayDuration = delayDuration
        self.bubbleTimerDuration = bubbleTimerDuration

        # Set up the rest of the window
        self.setFixedSize(400, 450)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.view = QGraphicsView(self)
        self.view.setGeometry(-100, -50, 500, 450)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setStyleSheet("background: transparent; border: none;")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        # Initialize the sprite item with the current outfit and expression
        self.initSpriteItem()

        # Load the config and conversation history
        self.configFile = 'config.json'
        self.loadConfig()
        self.loadConversationHistory()

        # Position the window at the bottom right of the screen
        screenGeometry = QApplication.desktop().screenGeometry()
        x = screenGeometry.width() - self.width()
        y = screenGeometry.height() - self.height()
        self.move(x, y)

        # Load and position the speech bubble sprite
        speechBubblePixmap = QPixmap('sprites/bubble.png')
        self.speechBubbleItem = QGraphicsPixmapItem(speechBubblePixmap)
        self.speechBubbleItem.setPos(-100, -100)
        self.scene.addItem(self.speechBubbleItem)

        # Initialize and position the message QLabel
        self.messageLabel = QLabel(self)
        self.messageLabel.setFont(customFont)
        self.messageLabel.setAlignment(Qt.AlignCenter)

        # Adjust these values to move the message label
        xPosition = 10
        yPosition = 10
        labelWidth = 100
        labelHeight = 180
        self.messageLabel.setGeometry(
            xPosition, yPosition, labelWidth, labelHeight)
        self.messageLabel.setWordWrap(True)

        # Create a QTimer for hiding the speech bubble
        self.hideBubbleTimer = QTimer(self)
        self.hideBubbleTimer.timeout.connect(self.hideSpeechBubble)

        # The timer should work only once per activation
        self.hideBubbleTimer.setSingleShot(True)

        # Initially, the speech bubble is hidden
        self.speechBubbleItem.setVisible(False)

        # Add a chat box
        self.chatBox = QLineEdit(self)
        self.chatBox.setGeometry(10, 400, 380, 25)
        self.chatBox.setStyleSheet(
            "background-color: white; color: black; border: 2px solid black; border-radius: 5px;")
        self.chatBox.setPlaceholderText("Use the 'gpt:' prefix for ChatGPT.")
        self.chatBox.setFocus()
        self.chatBox.returnPressed.connect(self.processCommand)

        # Connect the custom signal to the slot
        self.displayDelayedResponse.connect(self.updateDisplayForTTS)

        self.currentPromptType = self.loadConfig()
        self.createDatabase(self.currentPromptType)
        self.loadConversationHistory()

    def createDatabase(self, promptType):
        tableName = f'history_{promptType}'
        try:
            self.dbCursor.execute(f'''CREATE TABLE IF NOT EXISTS {tableName}
                                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    role TEXT,
                                    content TEXT,
                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
            self.dbConnection.commit()
        except sqlite3.Error as e:
            # Debugging message
            print(f"An error occurred while creating the table: {e}")

    def saveConversationHistory(self, entry=None):
        tableName = f'history_{self.currentPromptType}'
        if entry is not None:
            self.dbCursor.execute(f"INSERT INTO {tableName} (role, content) VALUES (?, ?)",
                                  (entry['role'], entry['content']))
        else:
            for entry in self.conversationHistory:
                self.dbCursor.execute(f"INSERT INTO {tableName} (role, content) VALUES (?, ?)",
                                      (entry['role'], entry['content']))
        self.dbConnection.commit()

    def loadConversationHistory(self):
        self.conversationHistory = []
        tableName = f'history_{self.currentPromptType}'
        try:
            for row in self.dbCursor.execute(f"SELECT role, content FROM {tableName} ORDER BY timestamp"):
                self.conversationHistory.append(
                    {"role": row[0], "content": row[1]})
        except sqlite3.OperationalError as e:
            # print(f"No such table: {tableName}. Error: {e}") # Debugging message
            # Attempt to create the table
            self.createDatabase(self.currentPromptType)

    def switchPrompt(self, newPromptType):
        self.currentPromptType = newPromptType
        # Create table for the new prompt if it doesn't exist
        self.createDatabase(newPromptType)
        self.loadConversationHistory()  # Load history from the new table
        self.saveConfig()

    def closeEvent(self, event):
        # Terminate the TTSWorker process if it's running
        if hasattr(self, 'ttsWorker'):
            try:
                if self.ttsWorker.isRunning():
                    self.ttsWorker.terminate()
                self.ttsWorker.process.join()  # Wait for the process to terminate
            except Exception as e:
                print(f"Error while terminating TTSWorker: {e}")

        # Terminate the GPTWorker thread if it's running
        if hasattr(self, 'gptWorker') and self.gptWorker.isRunning():
            self.gptWorker.terminate()
            self.gptWorker.wait()

        self.dbConnection.close()
        event.accept()

    def initSpriteItem(self):
        initialPixmap = self.sprites[self.currentOutfit][self.currentExpression]
        self.spriteItem = ClickablePixmapItem(initialPixmap, self)
        self.scene.addItem(self.spriteItem)

    def saveConfig(self):
        config = {
            'outfit': self.currentOutfit,
            'promptType': self.currentPromptType
        }
        with open(self.configFile, 'w') as f:
            json.dump(config, f, indent=4)

    def loadConfig(self):
        if os.path.exists(self.configFile):
            with open(self.configFile, 'r') as f:
                config = json.load(f)
                self.currentOutfit = config.get('outfit', 'default')
                self.currentPromptType = config.get('promptType', 'default')
                self.shortcuts = config.get('shortcuts', {})
                self.lights = config.get('lights', {})
                self.updateSprite()
                return config.get('promptType', 'default')
        else:
            self.currentOutfit = 'default'
            self.currentPromptType = 'default'
            self.shortcuts = {}
            self.lights = {}
            return self.currentPromptType

    def cycleOutfit(self):
        currentIndex = self.outfits.index(self.currentOutfit)
        newIndex = (currentIndex + 1) % len(self.outfits)
        self.currentOutfit = self.outfits[newIndex]
        self.updateSprite()
        self.saveConfig()

    def reverseCycleOutfit(self):
        currentIndex = self.outfits.index(self.currentOutfit)
        newIndex = (currentIndex - 1) % len(self.outfits)
        self.currentOutfit = self.outfits[newIndex]
        self.updateSprite()
        self.saveConfig()

    def changeExpression(self, newExpression):
        if newExpression in self.expressions:
            self.currentExpression = newExpression
            self.updateSprite()

    def updateSprite(self):
        newPixmap = self.sprites[self.currentOutfit][self.currentExpression]
        self.spriteItem.setPixmap(newPixmap)

    def blink(self):
        self.isBlinking = True
        self.blinkingIndex = 0
        self.blinkFrameTimer.start(self.blinkFrameSpeed)

    def blinkFrame(self, manualControl=False):
        if self.isBlinking or manualControl:
            blinkSequence = self.blinkingSprites[self.currentOutfit][self.currentExpression]

            if self.blinkingIndex < len(blinkSequence):
                self.spriteItem.setPixmap(blinkSequence[self.blinkingIndex])
                self.blinkingIndex += 1
            else:
                self.blinkingIndex = 0
                self.isBlinking = False
                self.blinkFrameTimer.stop() if not manualControl else None
                self.spriteItem.setPixmap(
                    self.sprites[self.currentOutfit][self.currentExpression])

    def testExpressions(self):
        for expression in self.expressions:
            self.changeExpression(expression)
            QApplication.processEvents()
            time.sleep(1)

            # Manually trigger and control the blink animation
            self.isBlinking = True
            for frameIndex in range(4):  # Iterate through frames 0 to 4
                self.blinkingIndex = frameIndex
                self.blinkFrame(manualControl=True)
                QApplication.processEvents()
                time.sleep(0.1)  # Adjust timing as needed

            # Reset to the normal state
            self.isBlinking = False
            self.blinkingIndex = 0
            self.spriteItem.setPixmap(
                self.sprites[self.currentOutfit][self.currentExpression])
            QApplication.processEvents()

    def hideSpeechBubble(self):
        self.speechBubbleItem.setVisible(False)
        self.messageLabel.clear()  # Clear the text from the message label

    def getHelpMessage(self):
        helpMessage = (
            "Available Commands:\n"
            "- volume\n"
            "- hide/show\n"
            "- pause/play\n"
            "- switch [prompt]\n"
        )
        return helpMessage

    def controlHomeAssistant(self, entity_id, action):
        headers = {
            "Authorization": f"Bearer {haToken}",
            "content-type": "application/json",
        }
        data = {"entity_id": entity_id}
        url = f"{haUrl}/api/services/light/{action}"
        try:
            response = requests.post(url, json=data, headers=headers)
            return response.ok
        except Exception as e:
            print(f"Error controlling Home Assistant device: {e}")
            return False

    def getHomeAssistantState(self, entity_id):
        headers = {
            "Authorization": f"Bearer {haToken}",
            "content-type": "application/json",
        }
        url = f"{haUrl}/api/states/{entity_id}"
        try:
            response = requests.get(url, headers=headers)
            if response.ok:
                return response.json()['state']
            else:
                return None
        except Exception as e:
            print(f"Error fetching state from Home Assistant: {e}")
            return None

    def toggleHomeAssistantLight(self, entity_id):
        current_state = self.getHomeAssistantState(entity_id)
        if current_state is None:
            print("Error getting current state")
            return False

        action = "turn_off" if current_state == "on" else "turn_on"
        return self.controlHomeAssistant(entity_id, action)

    def processCommand(self):
        command = self.chatBox.text().strip().lower()

        # Define a prefix for chatgpt
        prefix = "gpt:"

        if command == "quit" or command == "close" or command == "exit" or command == "q":
            self.close()
            return
        elif command == "volume up":
            volume.SetMasterVolumeLevelScalar(
                volume.GetMasterVolumeLevelScalar() + 0.1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chatBox.clear()
            return
        elif command == "volume down":
            volume.SetMasterVolumeLevelScalar(
                volume.GetMasterVolumeLevelScalar() - 0.1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chatBox.clear()
            return
        elif command == "volume mute" or command == "volume 0":
            volume.SetMasterVolumeLevelScalar(0, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chatBox.clear()
            return
        elif command == "volume max" or command == "volume 1":
            volume.SetMasterVolumeLevelScalar(1, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chatBox.clear()
            return
        elif command == "volume mid":
            volume.SetMasterVolumeLevelScalar(0.5, None)
            self.current = volume.GetMasterVolumeLevel()
            self.chatBox.clear()
            return
        elif command.startswith("volume "):
            try:
                percent = float(command.split("volume ")[1])
                if 0 <= percent <= 100:
                    # Convert percentage to a value between 0 and 1
                    volumeLevel = percent / 100.0
                    volume.SetMasterVolumeLevelScalar(volumeLevel, None)
                    self.current = volume.GetMasterVolumeLevel()
                    self.chatBox.clear()
                else:
                    self.chatBox.setText(
                        "Invalid volume percentage. Please use a value between 0 and 100.")
            except ValueError:
                self.chatBox.setText(
                    "Invalid volume percentage. Please use a numeric value.")
            return
        elif command == "hide":
            # Disable always on top
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
            self.setVisible(True)
            self.showMinimized()
            self.chatBox.clear()
            return
        elif command == "show":
            self.show()
            self.setWindowFlag(Qt.WindowStaysOnTopHint,
                               True)  # Enable always on top
            self.setVisible(True)
            self.chatBox.clear()
            return
        elif command == "pause" or command == "play" or command == "p":
            simulatePlayPause()
            self.chatBox.clear()
            return
        elif command.lower().startswith("switch "):
            newPromptType = command[len("switch "):].strip()
            self.switchPrompt(newPromptType)
            self.chatBox.clear()
        # elif command == "test":
        #     self.testExpressions()
        #     self.chatBox.clear()
        #     return
        elif command == "toggle":
            self.noTtsMode = not self.noTtsMode  # Toggle the TTS mode
            response = "TTS Mode Disabled" if self.noTtsMode else "TTS Mode Enabled"
            self.speechBubbleItem.setVisible(True)  # Show the speech bubble
            self.hideBubbleTimer.start(self.bubbleTimerDuration)
            self.messageLabel.setText(response)
            self.chatBox.clear()
            return
        # elif command == "message":
        #     defaultMessage = "This is the default message."
        #     self.speechBubbleItem.setVisible(True)  # Show the speech bubble
        #     self.hideBubbleTimer.start(self.bubbleTimerDuration)
        #     self.messageLabel.setText(defaultMessage)
        #     print("Default message set")  # Debugging message
        #     self.chatBox.clear()
        #     return
        elif command == "help":
            helpMessage = self.getHelpMessage()
            self.speechBubbleItem.setVisible(True)
            self.hideBubbleTimer.start(self.bubbleTimerDuration)
            self.messageLabel.setText(helpMessage)
            self.chatBox.clear()
            return
        elif command.startswith("play "):
            song_name = command[len("play "):].strip()
            # search_and_play(song_name)
            self.speechBubbleItem.setVisible(True)
            self.hideBubbleTimer.start(self.bubbleTimerDuration)
            self.messageLabel.setText(search_and_play(song_name))
            self.chatBox.clear()
        elif command in self.shortcuts:
            keyboard.send(self.shortcuts[command])
            self.chatBox.clear()
            return
        elif command.startswith("toggle"):
            light_name = command.replace("toggle", "").strip()
            if light_name in self.lights:
                self.toggleHomeAssistantLight(self.lights[light_name])
                self.chatBox.clear()
            else:
                print(f"Light '{light_name}' not found in configuration.")
            return

        # Check if the command starts with the prefix
        if not command.lower().startswith(prefix.lower()):
            self.chatBox.clear()
            return

        # Remove the prefix from the command
        command = command[len(prefix):].strip().lower()

        # Add user command to history
        newUserEntry = {"role": "user", "content": command}
        self.conversationHistory.append(newUserEntry)
        self.saveConversationHistory(newUserEntry)

        self.gptWorker = GPTWorker(
            command, self.conversationHistory, promptType=self.currentPromptType)
        self.gptWorker.finished.connect(self.handleGptResponse)
        self.gptWorker.start()

        self.chatBox.clear()

    def handleGptResponse(self, gptResponse):
        newAssistantEntry = {"role": "assistant", "content": gptResponse}
        self.conversationHistory.append(newAssistantEntry)
        self.saveConversationHistory(newAssistantEntry)

        if gptResponse.startswith('[') and ']' in gptResponse:
            endBracketIndex = gptResponse.find(']')
            # Convert expression to lowercase
            expression = gptResponse[1:endBracketIndex].strip().lower()
            message = gptResponse[endBracketIndex + 1:].strip()

            if expression in self.expressions:
                self.changeExpression(expression)
            else:
                message = gptResponse  # Use the original message if expression is not valid
        else:
            message = gptResponse  # Use the original message if no expression is found

        # Use threading to introduce delay
        threading.Thread(target=self.delayedDisplay, args=(message,)).start()

    def delayedDisplay(self, message):
        time.sleep(self.delayDuration)
        self.displayDelayedResponse.emit(message)

    def updateDisplayForTTS(self, message):
        # This method runs in the main thread
        self.messageLabel.setText(message)
        self.speechBubbleItem.setVisible(True)
        self.hideBubbleTimer.start(self.bubbleTimerDuration)

        if not self.noTtsMode:
            messages = [message]  # Prepare the message for TTS
            self.ttsWorker = TTSWorker(messages)
            self.ttsWorker.start()
        else:
            print(message)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    blinkSpeed = 25
    blinkTimer = 4000
    delayDuration = 1
    bubbleTimerDuration = 10000
    assistant = VirtualAssistant(
        blinkSpeed, blinkTimer, delayDuration, bubbleTimerDuration)
    assistant.show()
    sys.exit(app.exec_())
