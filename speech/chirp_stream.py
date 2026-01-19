import base64
import queue
import threading
from typing import Generator, Optional

from google.cloud import speech_v1 as speech


class ChirpStreamer:
    """
    Streaming speech-to-text client using Google Cloud Speech v2 (Chirp).
    
    Supports:
        - Real-time audio streaming via gRPC
        - Speaker diarization (labels speakers A, B, etc.)
        - Base64 audio input (16 kHz, 16-bit mono PCM)
    """
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
        """
        Add a base64-encoded audio chunk to the streaming queue.
        
        Args:
            b64: Base64-encoded audio data (16 kHz, 16-bit mono PCM).
        """
        try:
            self._audio_q.put(base64.b64decode(b64), block=False)
        except Exception:
            pass

    def finish(self) -> None:
        """Signal that no more audio will be sent; close the stream gracefully."""
        self._finished.set()

    def _get_streaming_config(self):
        """
        Create the streaming recognition config.
        
        Returns:
            speech.StreamingRecognitionConfig: The config for streaming recognition.
        """
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
        print(f"[DEBUG] Streaming config created: {streaming_config}")
        return streaming_config

    def _request_generator(self) -> Generator[speech.StreamingRecognizeRequest, None, None]:
        """
        Generate streaming requests for Google Speech gRPC API.
        
        Yields:
            speech.StreamingRecognizeRequest: Audio chunks.
        """
        audio_passed = False  # Track if any audio is passed
        while not self._finished.is_set() or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.1)
                if chunk:
                    audio_passed = True
                    print(f"[DEBUG] Sending audio chunk of size: {len(chunk)} bytes.")
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

        if not audio_passed:
            print("[DEBUG] Connection made but no audio was passed through.")

    def responses(self):
        """
        Start the streaming recognize call and return the response iterator.
        
        Returns:
            Iterable of speech.StreamingRecognizeResponse.
        """
        try:
            print("[DEBUG] Starting streaming recognize call (v1 API)...")
            streaming_config = self._get_streaming_config()
            response_iterator = self._client.streaming_recognize(streaming_config, self._request_generator())
            print("[DEBUG] Streaming recognize call started successfully.")
            return response_iterator
        except Exception as e:
            print(f"[ERROR] Error in streaming_recognize: {e}")
            raise


def speaker_label_from_result(result: speech.StreamingRecognitionResult) -> Optional[str]:
    """
    Extract a human-readable speaker label from a speech recognition result.
    
    Args:
        result: A single streaming recognition result with diarization.
    
    Returns:
        Speaker label like "Speaker A", "Speaker B", etc., or None if unavailable.
    """
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
