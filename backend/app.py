import json
import os
import tempfile
import threading
from typing import Optional

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_sock import Sock

from firebase.db import FirestoreDB
from speech.chirp_stream import ChirpStreamer, speaker_label_from_result
from asl.asl_inference import transcribe_video

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID")
db = FirestoreDB(project_id=FIREBASE_PROJECT_ID)
creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
if creds and creds.strip().startswith("{"):
    try:
        tmp_json = os.path.join(tempfile.gettempdir(), "gcp_creds.json")
        with open(tmp_json, "w") as f:
            f.write(creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_json
    except Exception:
        pass


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/speech/finalize")
def speech_finalize():
    data = request.get_json(silent=True) or {}
    conversation_id = data.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400
    try:
        db.finalize_conversation(conversation_id)
        return jsonify({"status": "finalized", "conversation_id": conversation_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/asl/transcribe")
def asl_transcribe():
    conversation_id = request.form.get("conversation_id")
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "video file is required (form field 'video')"}), 400

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        text = transcribe_video(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    try:
        if conversation_id and text:
            db.save_message(conversation_id=conversation_id, text=text, source="asl")
    except Exception:
        pass

    return jsonify({"text": text})


@sock.route("/speech/ws")
def speech_ws(ws):
    conversation_id: Optional[str] = request.args.get("conversation_id")
    streamer = ChirpStreamer()

    def consume_responses():
        try:
            for response in streamer.responses():
                for result in response.results:
                    if result.is_final:
                        try:
                            alt = result.alternatives[0]
                            transcript = (alt.transcript or "").strip()
                        except Exception:
                            transcript = ""
                        speaker = speaker_label_from_result(result)
                        if transcript:
                            try:
                                ws.send(json.dumps({
                                    "event": "final_transcript",
                                    "text": transcript,
                                    "speaker": speaker
                                }))
                            except Exception:
                                streamer.finish()
                                return
                            try:
                                if conversation_id:
                                    db.save_message(conversation_id=conversation_id, text=transcript, source="speech", speaker=speaker)
                            except Exception:
                                pass
        except Exception:
            pass

    t = threading.Thread(target=consume_responses, daemon=True)
    t.start()

    try:
        while True:
            msg = ws.receive()
            if msg is None:
                break
            try:
                data = json.loads(msg)
            except Exception:
                continue

            event = data.get("event")
            if event == "audio_chunk":
                b64 = data.get("data")
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                streamer.add_audio_base64(b64 or "")
            elif event in ("end", "finish", "close"):
                if not conversation_id:
                    cid = data.get("conversation_id")
                    if cid:
                        conversation_id = cid
                streamer.finish()
                break
            elif event == "set_conversation":
                cid = data.get("conversation_id")
                if cid:
                    conversation_id = cid
            else:
                pass
    except Exception:
        pass
    finally:
        try:
            streamer.finish()
        except Exception:
            pass
        try:
            t.join(timeout=2)
        except Exception:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
