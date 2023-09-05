import os
import base64
import bs4
import re
import logging

from datetime import datetime
from typing import List

from email.utils import parsedate_to_datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from .base import BaseDataloader, ChunkedDoc

from pendo.core import CREDENTIALS_GMAIL_PATH, TOKENS_PATH, WORKSPACE_PATH
from pendo.llms import get_llm, Message, MessageRole
from asyncio import Semaphore


GMAIL_TOKEN_PATH = TOKENS_PATH / "gmail_token.json"
GMAIL_PATH = WORKSPACE_PATH / "gmail"
GMAIL_TRACK_LIST = GMAIL_PATH / "track_list.txt"
GMAIL_IGNORE_LIST = GMAIL_PATH / "ignore_list.txt"

def _suppress_whitespace(text):
    if text is None:
        return None
    # Split the text by lines
    lines = text.split('\n')
    # For each line, split by whitespaces and then join
    suppressed_lines = [' '.join(line.split()) for line in lines]
    # Join the lines back with newlines
    text = "\n".join(suppressed_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text
    

def _html_to_text(html):
    soup = bs4.BeautifulSoup(html, "html.parser")
    return soup.get_text()

def _get_email_content(payload):
    mime_type = payload.get("mimeType", None)

    if mime_type == "multipart/alternative":
        for part in payload.get("parts", []):
            if part.get("mimeType", None) == "text/html":
                return _html_to_text(base64.urlsafe_b64decode(part.get("body", {}).get("data", "")).decode("utf-8"))
        for part in payload.get("parts", []):
            if part.get("mimeType", None) == "text/plain":
                return base64.urlsafe_b64decode(part.get("body", {}).get("data", "")).decode("utf-8")
    elif mime_type == "multipart/related" or mime_type == "multipart/mixed":
        return _get_email_content(payload.get("parts", [])[0])
    elif mime_type == "text/html":
        return _html_to_text(base64.urlsafe_b64decode(payload.get("body", {}).get("data", "")).decode("utf-8"))
    elif mime_type == "text/plain":
        return base64.urlsafe_b64decode(payload.get("body", {}).get("data", "")).decode("utf-8")
    
    return None

class GmailDataloader(BaseDataloader):

    def __init__(self, name, config, tokenizer):
        super().__init__(name, config, tokenizer)
        
        if not GMAIL_PATH.exists():
            GMAIL_PATH.mkdir(parents=True, exist_ok=True)
        if not GMAIL_TRACK_LIST.exists():
            GMAIL_TRACK_LIST.touch()
        if not GMAIL_IGNORE_LIST.exists():
            GMAIL_IGNORE_LIST.touch()

        self.track_list = self._load_list(GMAIL_TRACK_LIST)
        self.ignore_list = self._load_list(GMAIL_IGNORE_LIST)

        creds = None
        if os.path.exists(GMAIL_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH)
        
        if not creds or not creds.valid:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_GMAIL_PATH, ["https://www.googleapis.com/auth/gmail.readonly"])
                creds = flow.run_local_server(port=0)
            except Exception as e:
                logging.error(f"Unable to initiate Gmail client: {e}")
                raise e
            with open(GMAIL_TOKEN_PATH, "w") as token:
                token.write(creds.to_json())
        self.service = build("gmail", "v1", credentials=creds)
        self.llm = get_llm(config.get("llm", None))
        self.llm_semaphore = Semaphore(config.get("llm_coroutines", 20))
        assert self.llm is not None, f"GmailDataloader {self.name} - LLM is not configured"

    def _load_list(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def _add_to_list(self, path, item):
        with open(path, "a") as f:
            f.write(f"{item}\n")

    async def is_newsletter(self, subject, source, content):
        async with self.llm_semaphore:
            prompt = "Read the following paragraphs of the email, decide whether or not it is a newsletter. Beware that it should not be a promotional email, although sometimes it is ok to have ads. It should give useful information or knowledge about the world. It can be either news, compilation of news, opinions, substack, rss feed, or blog posts. If it is a newsletter, return `true`, otherwise, return `false`."
            result, usage = self.llm.chat_completion(
                    messages = [Message(MessageRole.SYSTEM, prompt), Message(MessageRole.USER, "\n".join([subject, source, content]))],
                    temperature = 0.6,
            )
        return "true" in result.content == "true"

    async def retrieve_doc_ids(self, after: datetime = None) -> List[str]:
        if after is None:
            after = self.get_timestamp()

        all_emails = []
        request = self.service.users().messages().list(userId="me", q=f'after:{int(after.timestamp())}', maxResults=500)

        while request is not None:
            response = request.execute()
            if "messages" in response:
                all_emails.extend(response.get("messages", []))
            
            request = self.service.users().messages().list_next(previous_request=request, previous_response=response)
        # check the pagination
        return [msg["id"] for msg in all_emails]

    async def retrieve_chunked_doc(self, doc_id: str) -> ChunkedDoc:
        async with self.semaphore:
            msg = self.service.users().messages().get(userId='me', id=doc_id).execute()
        headers = msg['payload']['headers']

        from_header = next((header for header in headers if header["name"] == "From"), None).get("value", "N/A")
        subject_header = next((header for header in headers if header["name"] == "Subject"), None).get("value", "N/A")
        date_header = next((header for header in headers if header["name"] == "Date"), None).get("value", None)
        if date_header:
            date_header = parsedate_to_datetime(date_header)

        content = _suppress_whitespace(_get_email_content(msg.get("payload", {})))
        if content is None:
            return None
        
        if from_header not in self.track_list and from_header not in self.ignore_list:
            is_newsletter = await self.is_newsletter(subject_header, from_header, content[:1000])
            if is_newsletter:
                self._add_to_list(GMAIL_TRACK_LIST, from_header)
                self.track_list.append(from_header)
            else:
                self._add_to_list(GMAIL_IGNORE_LIST, from_header)
                self.ignore_list.append(from_header)

        if from_header in self.ignore_list:
            return None

        return ChunkedDoc(
            id=doc_id,
            title=subject_header,
            last_edited_time=date_header.strftime("%Y-%m-%d %H:%M:%S"),
            chunks=[content],
            metadata={
                "source": from_header,
            }
        )
