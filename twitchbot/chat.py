import chat_downloader
import pandas as pd
import logging

log = logging.getLogger(__name__)


def download_chat(url: str):
    chat = chat_downloader.ChatDownloader()
    data = chat.get_chat(url)

    log.info(f"Chat downloaded from {url}!")

    pddata = pd.DataFrame(filter_chat(data))

    log.info("Chat loaded")

    pddata.to_csv("chat.csv", index=False)

    log.info("Chat saved to chat.csv!")

    return pddata


def filter_chat(chat):
    for message in chat:

        if message["message"] is None:
            continue

        if message["author"] is None:
            continue

        new_message = {}

        # new_message["message_id"] = message['message_id']
        new_message["timestamp"] = message["timestamp"]
        new_message["message"] = message["message"]
        # new_message["author_id"] = message['author']['id']

        yield new_message

    log.info("Chat saved to chat.csv!")


def load_chat():
    return pd.read_csv("chat.csv")
