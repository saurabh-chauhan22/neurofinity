import pyaudio
import wave

from loguru import logger

class AudioRecorder:
    '''
    Class to record audio from microphone
    '''
    def __init__(self):
        '''
        constructor for audio recorder
        '''
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.frames = []

    def record(self):
        '''
        record audio from microphone
        '''
        try:
            while True:
                data = self.stream.read(1024)
                self.frames.append(data)
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt: Recording stopped")
            pass
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            logger.info("Audio recording stopped")
            
    def save_audio(self):
        '''
        save audio to file
        '''
        sound_file = wave.open("recording.wav", "wb")
        sound_file.setnchannels(1)
        sound_file.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(44100)
        sound_file.writeframes(b''.join(self.frames))
        sound_file.close()
        logger.info("Audio saved to file")
        return sound_file
    
    def get_audio(self):
        '''
        get audio from file
        '''
        return wave.open("recording.wav", "rb")