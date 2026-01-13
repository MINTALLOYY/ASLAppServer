import os
from typing import Optional
from google.cloud import firestore


class FirestoreDB:
    def __init__(self, project_id: Optional[str] = None):
        self.client = firestore.Client(project=project_id)

    def save_message(self, conversation_id: str, text: str, source: str, speaker: Optional[str] = None):
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        conv_ref.set({"updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
        msg_ref = conv_ref.collection("messages").document()
        payload = {
            "text": text,
            "type": source,
            "speaker": speaker,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        msg_ref.set(payload)

    def finalize_conversation(self, conversation_id: str):
        if not conversation_id:
            return
        conv_ref = self.client.collection("conversations").document(conversation_id)
        conv_ref.set({"status": "finalized", "updated_at": firestore.SERVER_TIMESTAMP}, merge=True)
