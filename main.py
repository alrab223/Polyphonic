import os
import traceback

import discord
from discord.ext import commands
from dotenv import load_dotenv

# 環境変数を使うための設定
base_path = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(base_path, '.env')
load_dotenv(dotenv_path)


class MyBot(commands.Bot):

   def __init__(self, command_prefix, intents):

      super().__init__(command_prefix, intents=intents)
      for filename in os.listdir('cog'):
         if filename.endswith('.py'):
            try:
               self.load_extension(f'cog.{filename[:-3]}')
            except Exception:
               traceback.print_exc()


if __name__ == '__main__':
   intents = discord.Intents.all()
   bot = MyBot(command_prefix='$', intents=intents)
   bot.run(os.getenv("BOT"))
