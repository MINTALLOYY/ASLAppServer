import unittest
from google.cloud import speech_v1 as speech
from speech.chirp_stream import ChirpStreamer

class TestRealGRPCConnection(unittest.TestCase):

    def test_real_speech_client_initialization(self):
        """
        Test the real SpeechClient initialization to reproduce the error.
        """
        print("\n[TEST] Testing real SpeechClient initialization...")
        
        try:
            # Initialize a real ChirpStreamer (not mocked)
            streamer = ChirpStreamer(language_code="en-US", sample_rate_hz=16000, diarization_speaker_count=2)
            print("[TEST] ChirpStreamer initialized successfully.")
            
            # Try to call responses() without any audio
            # This should trigger the gRPC call
            print("[TEST] Attempting to call responses()...")
            responses = streamer.responses()
            
            # Try to iterate to trigger the actual gRPC connection
            print("[TEST] Attempting to consume responses...")
            for response in responses:
                print(f"[TEST] Received response: {response}")
                break  # Only check the first response
            
            print("[TEST] Real gRPC connection successful!")
            
        except Exception as e:
            print(f"[TEST] Error during real gRPC connection: {e}")
            print(f"[TEST] Error type: {type(e).__name__}")
            # Re-raise to fail the test
            raise

if __name__ == "__main__":
    unittest.main()
