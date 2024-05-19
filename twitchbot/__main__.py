import logging
from dotenv import load_dotenv
import os

from twitchbot.bot import Bot
import twitchbot.chat as chat
import twitchbot.sentiment_anal as sa
from twitchbot.keyword_analysis import KeywordAnalysis

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def main() -> None:

    # load_config()

    # log.debug("Starting bot")
    # access_token = os.getenv("TWITCH_ACCESS_TOKEN")
    # prefix = "?"
    # initial_channels = ("floppypidgen",)

    # bot = Bot(access_token, prefix, initial_channels)
    # bot.run()

    chat.download_chat("https://www.twitch.tv/videos/2121051911")

    data = sa.load_chat()

    counter = KeywordAnalysis()
    counter.fit(data["message"])

    print(counter.get_n_associated_keywords("bald", 10))

    print(counter.find_top_associated_keywords())
    pass


def load_config() -> None:
    log.debug("Loading config")
    load_dotenv()


if __name__ == "__main__":
    main()
