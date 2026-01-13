import base64
import queue
import threading
from typing import Generator, Optional

from google.cloud import speech_v1 as speech


class ChirpStreamer:
    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        diarization_speaker_count: int = 2,
        model: str = "chirp",
    ) -> None:
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.diarization_speaker_count = diarization_speaker_count
        self.model = model

        self._client = speech.SpeechClient()
        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._finished = threading.Event()

    def add_audio_base64(self, b64: str) -> None:
        try:
            self._audio_q.put(base64.b64decode(b64), block=False)
        except Exception:
            pass

    def finish(self) -> None:
        self._finished.set()

    def _request_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        diarization_config = speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=max(1, self.diarization_speaker_count),
        )
        rec_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=True,
            diarization_config=diarization_config,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=rec_config,
            interim_results=False,
            single_utterance=False,
        )
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)

        while not self._finished.is_set() or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.1)
                if chunk:
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def responses(self):
        return self._client.streaming_recognize(requests=self._request_generator())


def speaker_label_from_result(result: speech.StreamingRecognitionResult) -> Optional[str]:
    try:
        alt = result.alternatives[0]
        if alt.words:
            tag = alt.words[-1].speaker_tag
            if tag:
                base = ord('A') - 1
                return f"Speaker {chr(base + int(tag))}"
    except Exception:
        pass
    return None
