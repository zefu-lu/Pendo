import os
import base64

from email.utils import parsedate_to_datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from .base import BaseDataloader, ChunkedDoc
from datetime import datetime
from typing import Dict, List
from core.paths import CREDENTIALS_GMAIL_PATH, TOKENS_PATH

import logging

GMAIL_TOKEN_PATH = TOKENS_PATH / "gmail_token.json"

def _get_email_content(msg):
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            mime_type = part['mimeType']
            if mime_type == 'text/plain':
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
            elif mime_type == 'text/html':
                # If you prefer HTML content over plain text, you can adjust the priority here
                return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    else:
        return base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
    return None

class GmailDataloader(BaseDataloader):

    def __init__(self, name, config, tokenizer):
        super().__init__(name, config, tokenizer)
        
        creds = None
        if os.path.exists(GMAIL_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_GMAIL_PATH, ["https://www.googleapis.com/auth/gmail.readonly"])
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    logging.error(f"Unable to initiate Gmail client: {e}")
                    raise e
            with open(GMAIL_TOKEN_PATH, "w") as token:
                token.write(creds.to_json())
        self.service = build("gmail", "v1", credentials=creds)


    async def retrieve_doc_ids(self, after: datetime = None) -> List[str]:
        results = self.service.users().messages().list(userId="me", maxResults=10).execute()
        messages = results.get("messages", [])
        return [msg["id"] for msg in messages]

    async def retrieve_chunked_doc(self, doc_id: str) -> ChunkedDoc:
        msg = self.service.users().messages().get(userId='me', id=doc_id).execute()
        headers = msg['payload']['headers']

        from_header = next((header for header in headers if header["name"] == "From"), None).get("value", "N/A")
        subject_header = next((header for header in headers if header["name"] == "Subject"), None).get("value", "N/A")
        date_header = next((header for header in headers if header["name"] == "Date"), None).get("value", None)
        if date_header:
            date_header = parsedate_to_datetime(date_header)

        content = _get_email_content(msg)
        return ChunkedDoc(
            id=doc_id,
            title=subject_header,
            last_edited_time=date_header,
            chunks=[content]
        )
