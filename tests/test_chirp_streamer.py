import unittest
from unittest.mock import MagicMock, patch
from google.cloud import speech_v1 as speech
from speech.chirp_stream import ChirpStreamer

class TestChirpStreamer(unittest.TestCase):

    @patch("google.cloud.speech_v1.SpeechClient")
    def test_request_generator(self, MockSpeechClient):
        # Mock the SpeechClient
        mock_client = MockSpeechClient.return_value

        # Initialize ChirpStreamer
        streamer = ChirpStreamer(language_code="en-US", sample_rate_hz=16000, diarization_speaker_count=2)

        # Mock audio data
        audio_data = b"test_audio_data"
        streamer._audio_q.put(audio_data)
        streamer._finished.set()

        # Get the generator
        generator = streamer._request_generator()

        # First request should contain the streaming config
        first_request = next(generator)
        self.assertIsInstance(first_request, speech.StreamingRecognizeRequest)
        self.assertIsNotNone(first_request.streaming_config)
        self.assertIsInstance(first_request.streaming_config, speech.StreamingRecognitionConfig)

        # Second request should contain audio data
        second_request = next(generator)
        self.assertIsInstance(second_request, speech.StreamingRecognizeRequest)
        self.assertEqual(second_request.audio_content, audio_data)

    @patch("google.cloud.speech_v1.SpeechClient")
    def test_responses(self, MockSpeechClient):
        # Mock the SpeechClient
        mock_client = MockSpeechClient.return_value
        mock_response = MagicMock()
        mock_client.streaming_recognize.return_value = iter([mock_response])

        # Initialize ChirpStreamer
        streamer = ChirpStreamer(language_code="en-US", sample_rate_hz=16000, diarization_speaker_count=2)

        # Mock the _request_generator
        streamer._request_generator = MagicMock(return_value=iter([]))

        # Call responses and check the result
        responses = streamer.responses()
        self.assertEqual(list(responses), [mock_response])
        mock_client.streaming_recognize.assert_called_once_with(requests=streamer._request_generator())

if __name__ == "__main__":
    unittest.main()