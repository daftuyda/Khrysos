from RealtimeTTS import TextToAudioStream, ElevenlabsEngine
from dotenv import load_dotenv
import os

load_dotenv()


def dummy_generator():
    yield "Hey guys! "
    yield "These here are "
    yield "realtime spoken words "
    yield "based on eleven labs "
    yield "TTS text synthesis."


engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
stream = TextToAudioStream(engine)
stream.feed(dummy_generator())

print("Synthesizing...")
stream.play()
