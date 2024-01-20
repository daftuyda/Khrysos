import sys
from cx_Freeze import setup, Executable

sys.setrecursionlimit(2000)

# Dependencies are automatically detected, but some modules need help.
build_exe_options = {
    "packages": [
        "os",
        "sys",
        "ctypes",
        "json",
        "sqlite3",
        "requests",
        "keyboard",
        "sympy",
        "spotipy",
        "whisper",
        "torch",
        "string",
        "time",
        "socket",
        "re",
        "random",
        "numpy",
        "speech_recognition",
        "pypresence",
        "queue",
        "ctransformers",
        "dotenv",
        "openai",
        "PyQt5.QtMultimedia",
        "PyQt5.QtWidgets",
        "PyQt5.QtGui",
        "PyQt5.QtCore",
    ],
    "excludes": [],
    "include_files": [
        ".env",
        "config.json",
        "conversationHistory.db",
        "font/",
        "prompts/",
        "sounds/",
        "sprites/",
        "static/",
        "subtitles.txt",
    ],
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Khrysos",
    version="0.1",
    description="An AI Desktop Assistant",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base)],
)
