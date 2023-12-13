import sys
import os
import time
import ctypes
import json
import sqlite3
import requests
import keyboard
import sympy
import spotipy
import speech_recognition as sr
from spotipy.oauth2 import SpotifyOAuth
from pytube import YouTube
from multiprocessing import Process, Queue
from elevenlabs import generate, play, Voice, VoiceSettings, set_api_key
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume
from dotenv import load_dotenv
from openai import OpenAI
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QLineEdit, QLabel
from PyQt5.QtGui import QPixmap, QFont, QFontDatabase
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QObject

load_dotenv()

openWeatherKey = os.getenv('openWeatherApi')

haUrl = os.getenv('haUrl')
haToken = os.getenv('haToken')

clientId = os.getenv('SpotifyClientId')
secretId = os.getenv('SpotifySecretId')
redirectUri = os.getenv('SpotifyRedirectUri')
scope = "user-read-private user-read-playback-state user-modify-playback-state"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=clientId,
                                               client_secret=secretId,
                                               redirect_uri=redirectUri,
                                               scope=scope))


def searchAndPlay(songName):
    results = sp.search(q=songName, limit=1)
    if results['tracks']['items']:
        song_uri = results['tracks']['items'][0]['uri']
        sp.start_playback(uris=[song_uri])
        return f"Playing: {results['tracks']['items'][0]['name']}"
    else:
        return "Song not found."


def getPlaylists():
    playlists = sp.current_user_playlists(limit=10)
    playlist_info = ""
    for playlist in playlists['items']:
        playlist_info += f"{playlist['name']}\n"
    return playlist_info


def getPlaylistId(sp, playlistName):
    playlists = sp.current_user_playlists()
    for playlist in playlists['items']:
        if playlist['name'].lower() == playlistName.lower():
            return playlist['id']
    return None


def playPlaylist(playlistName):
    playlistId = getPlaylistId(sp, playlistName)
    if playlistId:
        sp.start_playback(context_uri=f'spotify:playlist:{playlistId}')
    else:
        return ("Playlist not found.")


def queueSong(songName):
    results = sp.search(q=songName, limit=1)
    if results['tracks']['items']:
        song_uri = results['tracks']['items'][0]['uri']
        sp.add_to_queue(song_uri)
        return f"Queued: {results['tracks']['items'][0]['name']}"
    else:
        return "Song not found."


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Constants for the Play/Pause key
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_EXTENDEDKEY = 0x1
KEYEVENTF_KEYUP = 0x2

user32 = ctypes.WinDLL('user32', use_last_error=True)


def keybd_event(bVk, bScan, dwFlags, dwExtraInfo):
    user32.keybd_event(bVk, bScan, dwFlags, dwExtraInfo)


def simulatePlayPause():
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0, KEYEVENTF_EXTENDEDKEY, 0)
    keybd_event(VK_MEDIA_PLAY_PAUSE, 0,
                KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)


def getAppIdByName(processName):
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        if session.Process:
            # Compare without '.exe' and case-insensitive
            sessionProcessName = session.Process.name().lower()
            if processName.lower() in sessionProcessName:
                return session.ProcessId
    return None


def setAppVolumeByName(processName, level):
    processId = getAppIdByName(processName)
    if processId is not None:
        setAppVolume(processId, level)
    else:
        return (f"No process found with the name '{processName}'")


def setAppVolume(processId, level):
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        if session.Process and session.ProcessId == processId:
            volume = session._ctl.QueryInterface(ISimpleAudioVolume)
            volume.SetMasterVolume(level, None)


def listAudioSessions():
    sessions = AudioUtilities.GetAllSessions()
    sessionInfo = ""
    for session in sessions:
        if session.Process:
            processName = session.Process.name()
            # Remove '.exe' from the process name if present
            if processName.lower().endswith(".exe"):
                processName = processName[:-4]
            sessionInfo += f"{processName}\n"
        else:
            pass
    return sessionInfo


def calculateExpr(expression):
    try:
        # Evaluate the expression using sympy
        result = sympy.sympify(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def get_weather(cityName, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={cityName}"
    response = requests.get(complete_url)
    return response.json()


def ttsTask(texts, apiKey, queue):
    try:
        set_api_key(apiKey)
        for text in texts:
            audio = generate(
                text=text,
                voice=Voice(
                    voice_id='EqudVpb9UKv174PwNcST',
                    settings=VoiceSettings(
                        stability=0.25, similarity_boost=0.3, style=0.0, use_speaker_boost=True)
                ),
                model="eleven_turbo_v2"
            )
            play(audio)

    except Exception as e:
        queue.put(str(e))
    finally:
        queue.put("done")


openAiKey = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openAiKey)


class DownloadWorker(QObject):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, url, path='downloads/'):
        super().__init__()
        self.url = url
        self.path = path

    def run(self):
        try:
            yt = YouTube(self.url)
            video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            if video:
                video.download(self.path)
                self.finished.emit(f"Downloaded: {yt.title}")
            else:
                self.error.emit("No suitable video stream found.")
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


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


class ContinuousSpeechRecognition(QThread):
    recognizedText = pyqtSignal(str)
    recognizedCommand = pyqtSignal(str)

    def __init__(self):
        super(ContinuousSpeechRecognition, self).__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_flag = False

    def run(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            self.recognizer.dynamic_energy_threshold = False
            self.recognizer.energy_threshold = 300

        while not self.stop_flag:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(
                        source, timeout=0.5)

                text = self.recognizer.recognize_google(audio)
                if text.strip():
                    # print(text)  # Debugging message
                    self.recognizedText.emit(text)
                    self.recognizedCommand.emit(text)

            except sr.WaitTimeoutError:
                # Timeout occurs, loop continues
                pass
            except sr.UnknownValueError:
                # Could not understand audio
                pass
            except sr.RequestError as e:
                # Could not request results
                pass

    def startRecognition(self):
        self.stop_flag = False
        self.start()

    def stopRecognition(self):
        self.stop_flag = True


class ClickableBox(QMainWindow):
    boxClicked = pyqtSignal()

    def __init__(self, parent=None):
        super(ClickableBox, self).__init__(parent)
        self.setFixedSize(25, 25)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(
            "background-color: rgba(0, 0, 0, 0.75); border-radius: 5px;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.boxClicked.emit()
            self.hide()


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
    def __init__(self, blinkSpeed=35, blinkTimer=4000, bubbleTimerDuration=10000):
        super().__init__()

        self.speech_recognition_thread = ContinuousSpeechRecognition()
        self.speech_recognition_thread.recognizedText.connect(
            self.processRecognizedText)
        self.speech_recognition_thread.recognizedCommand.connect(
            self.processVoiceCommand)
        self.speech_recognition_thread.start()

        self.dbConnection = sqlite3.connect('conversationHistory.db')
        self.dbCursor = self.dbConnection.cursor()

        self.currentPromptType = "default"
        self.currentOutfit = 'default'
        self.currentExpression = 'normal'
        self.isBlinking = False
        self.blinkingIndex = 0
        self.noTtsMode = False
        self.isAppVisible = True

        # Initialize and connect the ClickableBox
        self.clickableBox = ClickableBox(self)
        self.clickableBox.boxClicked.connect(self.showComponents)
        self.clickableBox.hide()  # Initially hidden

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
            return ("Failed to load font")
        else:
            fontFamily = QFontDatabase.applicationFontFamilies(fontId)[0]
            customFont = QFont(fontFamily)
            customFont.setPointSize(6)
            customFont.setLetterSpacing(QFont.PercentageSpacing, 90)

        # Timer for blinking animation
        self.blinkTimer = QTimer(self)
        self.blinkTimer.timeout.connect(self.blink)
        self.blinkTimer.start(blinkTimer)

        # Timer for resetting expression
        self.resetExpressionTimer = QTimer(self)
        self.resetExpressionTimer.timeout.connect(self.resetExpression)
        self.resetExpressionDuration = 15000

        # Timer for blinking animation frames
        self.blinkFrameTimer = QTimer(self)
        self.blinkFrameTimer.timeout.connect(self.blinkFrame)
        self.blinkFrameSpeed = blinkSpeed

        # Set bubble timer durations
        self.bubbleTimerDuration = bubbleTimerDuration

        # Set up the rest of the window
        self.setFixedSize(400, 430)
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
        self.chatBox.setPlaceholderText("Use the 'gpt' prefix for ChatGPT.")
        self.chatBox.setFocus()
        self.chatBox.returnPressed.connect(self.processChatboxCommand)

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
            return (f"An error occurred while creating the table: {e}")

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

    def hideWindow(self):
        self.isAppVisible = False

        # Hide the main window components
        self.spriteItem.setVisible(False)
        self.speechBubbleItem.setVisible(False)
        self.messageLabel.setVisible(False)
        self.chatBox.setVisible(False)

        self.clickableBox.raise_()  # Bring the box to the front
        self.clickableBox.setFocus()  # Set focus to the box
        self.clickableBox.move(350, 380)  # Hard code position cus I suck
        self.clickableBox.show()

    def showComponents(self):
        self.isAppVisible = True

        # Show the main application components
        self.spriteItem.setVisible(True)
        self.chatBox.setVisible(True)
        self.messageLabel.setVisible(True)

    def forceCloseApplication(self):
        # Terminate the speech recognition thread
        if hasattr(self, 'speech_recognition_thread') and self.speech_recognition_thread.isRunning():
            self.speech_recognition_thread.stopRecognition()
            self.speech_recognition_thread.terminate()
            self.speech_recognition_thread.wait()

        # Terminate the TTSWorker process
        if hasattr(self, 'ttsWorker') and self.ttsWorker.isRunning():
            self.ttsWorker.terminate()
            self.ttsWorker.process.join()

        # Terminate the GPTWorker thread
        if hasattr(self, 'gptWorker') and self.gptWorker.isRunning():
            self.gptWorker.terminate()
            self.gptWorker.wait()

        # Close database connection
        if hasattr(self, 'dbConnection'):
            self.dbConnection.close()

        # Close the application
        self.close()

    def initSpriteItem(self):
        initialPixmap = self.sprites[self.currentOutfit][self.currentExpression]
        self.spriteItem = ClickablePixmapItem(initialPixmap, self)
        self.scene.addItem(self.spriteItem)

    def startTimer(self, seconds):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.onTimerComplete)
        self.timer.start(seconds * 1000)  # QTimer takes milliseconds

    def onTimerComplete(self):
        self.timer.stop()
        self.speechBubbleItem.setVisible(True)
        self.hideBubbleTimer.start(self.bubbleTimerDuration)
        self.messageLabel.setText("Timer completed!")
        QSound.play("sounds/timer.wav")

    def saveConfig(self):
        config = {
            'outfit': self.currentOutfit,
            'promptType': self.currentPromptType,
            'keyword': self.keyword,
            'shortcuts': self.shortcuts,
            'lights': self.lights
        }
        with open(self.configFile, 'w') as f:
            json.dump(config, f, indent=4)

    def loadConfig(self):
        if os.path.exists(self.configFile):
            with open(self.configFile, 'r') as f:
                config = json.load(f)
                self.currentOutfit = config.get('outfit', 'default')
                self.currentPromptType = config.get('promptType', 'default')
                self.keyword = config.get('keyword', 'hey')
                self.shortcuts = config.get('shortcuts', {})
                self.lights = config.get('lights', {})
                self.updateSprite()
                return config.get('promptType', 'default')
        else:
            self.currentOutfit = 'default'
            self.currentPromptType = 'default'
            self.keyword = 'hey'
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
        if newExpression in self.expressions and newExpression != self.currentExpression:
            self.currentExpression = newExpression
            self.updateSprite()
            if newExpression != 'normal':  # Assuming 'normal' is your default expression
                self.resetExpressionTimer.start(self.resetExpressionDuration)

    def resetExpression(self):
        self.changeExpression('normal')  # Reset to default expression
        self.resetExpressionTimer.stop()  # Stop the timer

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

    def controlHomeAssistant(self, entityId, action):
        headers = {
            "Authorization": f"Bearer {haToken}",
            "content-type": "application/json",
        }
        data = {"entity_id": entityId}
        url = f"{haUrl}/api/services/light/{action}"
        try:
            response = requests.post(url, json=data, headers=headers)
            return response.ok
        except Exception as e:
            return (f"Error controlling Home Assistant device: {e}")

    def getHomeAssistantState(self, entityId):
        headers = {
            "Authorization": f"Bearer {haToken}",
            "content-type": "application/json",
        }
        url = f"{haUrl}/api/states/{entityId}"
        try:
            response = requests.get(url, headers=headers)
            if response.ok:
                return response.json()['state']
            else:
                return None
        except Exception as e:
            return (f"Error fetching state from Home Assistant: {e}")

    def toggleHomeAssistantLight(self, entityId):
        currentState = self.getHomeAssistantState(entityId)
        if currentState is None:
            return ("Error getting current state")

        action = "turn_off" if currentState == "on" else "turn_on"
        return self.controlHomeAssistant(entityId, action)

    def formatWeatherData(self, weatherData):
        try:
            city = weatherData['name']
            weatherCondition = weatherData['weather'][0]['description']
            temperature = weatherData['main']['temp']
            humidity = weatherData['main']['humidity']

            # Converting temperature from Kelvin to Celsius
            tempCelsius = temperature - 273.15

            # Formatting the message
            weatherMessage = (f"Weather in {city}:\n"
                              f"Condition: {weatherCondition.title()}\n"
                              f"Temperature: {tempCelsius:.2f}°C\n"
                              f"Humidity: {humidity}%")

            return weatherMessage
        except KeyError:
            return "Error: Could not parse weather data."

    def updateKeyword(self, newKeyword):
        self.keyword = newKeyword
        self.saveConfig()

    def processRecognizedText(self, text):
        # Check if the recognized text starts with the keyword
        if text.lower().startswith(self.keyword.lower()):
            # Remove the keyword from the text
            command = text[len(self.keyword):].strip()

            # Add user command to history
            newUserEntry = {"role": "user", "content": command}
            self.conversationHistory.append(newUserEntry)
            self.saveConversationHistory(newUserEntry)

            # Handle the command with GPT
            self.gptWorker = GPTWorker(
                command, self.conversationHistory, promptType=self.currentPromptType)
            self.gptWorker.finished.connect(self.handleGptResponse)
            self.gptWorker.start()
        else:
            pass

    def processVoiceCommand(self, command):
        # Check if the window is visible
        if not self.isAppVisible:
            return

        self.processCommand(command)

    def processChatboxCommand(self):
        # Get the text from the chatbox and process it as a command
        command = self.chatBox.text()
        self.processCommand(command)
        self.chatBox.clear()

    def processCommand(self, command):
        command = command.strip()

        # Define a prefix for chatgpt
        prefix = "gpt"

        if command.startswith("dl "):
            url = command[len("dl "):].strip()
            self.downloadWorker = DownloadWorker(url)
            # Connect signals
            self.downloadWorker.finished.connect(self.onDownloadFinished)
            self.downloadWorker.error.connect(self.onDownloadError)
            # Move the worker to a thread and start the thread
            self.downloadThread = QThread()
            self.downloadWorker.moveToThread(self.downloadThread)
            self.downloadThread.started.connect(self.downloadWorker.run)
            self.downloadThread.start()

            self.messageLabel.setText("Downloading video...")
            self.showBubble()

        command = command.lower()

        if command in ["quit", "close", "exit", "q"]:
            self.forceCloseApplication()
            return  # Return immediately to skip processing any further commands
        elif command in ["volume up", "vol up", "vol+"]:
            newVolumeLevel = volume.GetMasterVolumeLevelScalar() + 0.1
            if newVolumeLevel > 1:  # Ensure the volume level does not exceed 100%
                newVolumeLevel = 1
            volume.SetMasterVolumeLevelScalar(newVolumeLevel, None)
            # Convert to percentage and round off
            percentVolume = round(newVolumeLevel * 100)
            self.messageLabel.setText(f"Master volume set to {percentVolume}%")
            self.showBubble()
        elif command in ["volume down", "vol down", "vol-"]:
            newVolumeLevel = volume.GetMasterVolumeLevelScalar() - 0.1
            if newVolumeLevel < 0:  # Ensure the volume level does not go below 0%
                newVolumeLevel = 0
            volume.SetMasterVolumeLevelScalar(newVolumeLevel, None)
            # Convert to percentage and round off
            percentVolume = round(newVolumeLevel * 100)
            self.messageLabel.setText(f"Master volume set to {percentVolume}%")
            self.showBubble()
        elif command.startswith("volume ") or command.startswith("vol "):
            volumeCommand = command.split("volume ")[1].lower()
            if volumeCommand == "max":
                volumeLevel = 1.0
                percent = 100
            else:
                try:
                    percent = float(volumeCommand)
                    if 0 <= percent <= 100:
                        volumeLevel = percent / 100.0
                    else:
                        self.messageLabel.setText(
                            "Invalid volume percentage. Please use a value between 0 and 100.")
                        return
                except ValueError:
                    self.messageLabel.setText(
                        "Invalid volume percentage. Please use a numeric value or 'max'.")
                    return
            volume.SetMasterVolumeLevelScalar(volumeLevel, None)
            self.messageLabel.setText(f"Master volume set to {percent}%.")
            self.showBubble()
        elif (" volume ") in command or (" vol ") in command:
            parts = command.split()
            if len(parts) == 3:
                processName = parts[0]
                volumeCommand = parts[2]
                if volumeCommand.lower() == "max":
                    volumeLevel = 1.0
                    percent = 100
                else:
                    try:
                        percent = float(volumeCommand)
                        if 0 <= percent <= 100:
                            volumeLevel = percent / 100.0
                        else:
                            self.messageLabel.setText(
                                "Invalid volume percentage. Please use a value between 0 and 100.")
                            return
                    except ValueError:
                        self.messageLabel.setText(
                            "Invalid volume percentage. Please use a numeric value or 'max'.")
                        return

                setAppVolumeByName(processName, volumeLevel)
                self.messageLabel.setText(
                    f"{processName} volume set to {percent}%.")
            else:
                self.messageLabel.setText(
                    "Invalid command format. Use: [App name] volume [Level]")
            self.showBubble()
        elif command == "app list":
            self.messageLabel.setText(listAudioSessions())
            self.showBubble()
        elif command == "hide":
            self.hideWindow()
            self.chatBox.clear()
        elif command in ["pause", "play", "p"]:
            simulatePlayPause()
            self.messageLabel.setText("Toggled play/pause.")
            self.showBubble()
        elif command.lower().startswith("switch "):
            newPromptType = command[len("switch "):].strip()
            self.switchPrompt(newPromptType)
            self.messageLabel.setText(f"Switched to prompt: {newPromptType}")
            self.showBubble()
        elif command == "toggle":
            self.noTtsMode = not self.noTtsMode  # Toggle the TTS mode
            response = "TTS Mode Disabled" if self.noTtsMode else "TTS Mode Enabled"
            self.messageLabel.setText(response)
            self.showBubble()
        elif command == "help":
            helpMessage = self.getHelpMessage()
            self.messageLabel.setText(helpMessage)
            self.showBubble()
        elif command.startswith("play "):
            songName = command[len("play "):].strip()
            self.messageLabel.setText(queueSong(songName))
            self.showBubble()
        elif command in self.shortcuts:
            keyboard.send(self.shortcuts[command])
            self.messageLabel.setText(f"Shortcut: {command}")
            self.showBubble()
        elif command.startswith("toggle"):
            lightName = command.replace("toggle", "").strip()
            if lightName in self.lights:
                self.toggleHomeAssistantLight(self.lights[lightName])
                self.messageLabel.setText(f"Toggled light '{lightName}'.")
            else:
                self.messageLabel.setText(
                    f"Light '{lightName}' not found in configuration.")
            self.showBubble()
        elif command.startswith("playlist -l"):
            self.messageLabel.setText(getPlaylists())
            self.showBubble()
        elif command.startswith("playlist "):
            playlistId = command[len("playlist "):].strip()
            self.messageLabel.setText(playPlaylist(playlistId))
            self.showBubble()
        elif command in ["skip", "next", "n"]:
            sp.next_track()
            self.messageLabel.setText("Skipped to next track.")
            self.showBubble()
        elif command in ["back", "prev", "b"]:
            sp.previous_track()
            self.messageLabel.setText("Skipped to previous track.")
            self.showBubble()
        elif command.startswith("calc "):
            expression = command[len("calc "):].strip()
            result = calculateExpr(expression)
            self.messageLabel.setText(result)
            self.showBubble()
        elif command.startswith("timer "):
            try:
                timeInput = command[len("timer "):].strip()
                if ':' in timeInput:
                    minutes, seconds = map(int, timeInput.split(':'))
                    timeInSeconds = minutes * 60 + seconds
                else:
                    timeInSeconds = int(timeInput)

                self.startTimer(timeInSeconds)
                self.messageLabel.setText(f"Timer set for {timeInput}.")
            except ValueError:
                self.messageLabel.setText(
                    "Invalid time format. Please enter a number or time format (mm:ss).")
            self.showBubble()
        elif command.startswith("weather "):
            cityName = command[len("weather "):].strip()
            weatherData = get_weather(cityName, openWeatherKey)
            # Parse and format the weatherData as needed
            self.messageLabel.setText(self.formatWeatherData(weatherData))
            self.showBubble()
        elif command == "stt":
            if self.speech_recognition_thread.isRunning():
                self.speech_recognition_thread.stopRecognition()
                self.messageLabel.setText("Speech recognition turned off.")
            else:
                self.speech_recognition_thread.startRecognition()
                self.messageLabel.setText("Speech recognition turned on.")
            self.showBubble()
        elif command.startswith("keyword "):
            newKeyword = command[len("keyword "):].strip()
            self.updateKeyword(newKeyword)
            self.messageLabel.setText(f"Keyword set to: {newKeyword}")
            self.showBubble()

        # Check if the command starts with the prefix
        if command.lower().startswith(prefix.lower()):
            # Remove the prefix and process the GPT command
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
            expression = gptResponse[1:endBracketIndex].strip().lower()
            message = gptResponse[endBracketIndex + 1:].strip()

            if expression in self.expressions:
                self.changeExpression(expression)
            else:
                message = gptResponse  # Use the original message if expression is not valid
        else:
            message = gptResponse  # Use the original message if no expression is found

        self.startTTS(message)
        self.showBubble()
        self.messageLabel.setText(message)

    def showBubble(self):
        self.speechBubbleItem.setVisible(True)
        self.hideBubbleTimer.start(self.bubbleTimerDuration)
        self.chatBox.clear()

    def startTTS(self, text):
        if not self.noTtsMode:
            self.ttsWorker = TTSWorker([text])  # Pass the text as a list
            self.ttsWorker.start()

    def onDownloadFinished(self, message):
        # Update the UI with the success message
        self.messageLabel.setText(message)
        self.downloadThread.quit()
        self.downloadThread.wait()

    def onDownloadError(self, error_message):
        # Update the UI with the error message
        self.messageLabel.setText(error_message)
        self.downloadThread.quit()
        self.downloadThread.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    blinkSpeed = 25
    blinkTimer = 4000
    bubbleTimerDuration = 6000
    assistant = VirtualAssistant(
        blinkSpeed, blinkTimer, bubbleTimerDuration)
    assistant.show()
    sys.exit(app.exec_())
