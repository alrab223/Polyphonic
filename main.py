import glob
import os
import traceback

import discord
from discord.ext import commands
from dotenv import load_dotenv

cog_path = glob.glob("cog/*.py")
cog_path = [x.replace("/", ".") for x in cog_path]
cog_path = [x.replace(".py", "") for x in cog_path]
INITIAL_EXTENSIONS = cog_path


class MyBot(commands.Bot):

   def __init__(self, command_prefix, intents):

      super().__init__(command_prefix, intents=intents)

      for cog in INITIAL_EXTENSIONS:
         try:
            self.load_extension(cog)
         except Exception:
            traceback.print_exc()


if __name__ == '__main__':
   intents = discord.Intents.all()
   bot = MyBot(command_prefix='$', intents=intents)
   base_path = os.path.dirname(os.path.abspath(__file__))
   dotenv_path = os.path.join(base_path, '.env')
   load_dotenv(dotenv_path)
   bot.run(os.getenv("BOT"))
