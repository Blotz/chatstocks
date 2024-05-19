import logging
from twitchio.ext import commands

log = logging.getLogger(__name__)


class Bot(commands.Bot):

    def __init__(self, token: str, prefix: str, initial_channels: tuple[str]):
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels)

    async def event_ready(self):
        # Notify us when everything is ready!
        # We are logged in and ready to chat and use commands...
        log.info(f"Logged in as | {self.nick}")
        log.info(f"User id is | {self.user_id}")

        channel = self.get_channel("floppypidgen")
        await channel.send(f"Hello! I am now online in {channel.name}!")

    async def event_message(self, message):
        if message.author is None:
            return

        log.info(f"{message.author.name} | {message.content}")

        # Make the bot parse the commands
        await self.handle_commands(message)

    @commands.command()
    async def hello(self, ctx: commands.Context):
        # Send a hello back!
        # Sending a reply back to the channel is easy... Below is an example.
        log.info(f"{ctx.author.name} invoked the hello command!")

        await ctx.send(f"Hello {ctx.author.name}!")

        log.info(f"Replied with hello to {ctx.author.name}!")
