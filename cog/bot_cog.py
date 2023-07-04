import requests
from discord.ext import commands


class Main(commands.Cog):

   def __init__(self, bot):
      self.bot = bot
      self.pipe = None

   def download_img(self, url, file_name):
      r = requests.get(url, stream=True)
      if r.status_code == 200:
         with open(file_name, 'wb') as f:
            f.write(r.content)

   @commands.command("デリート")
   async def delete_bomb(self, ctx):
      await ctx.channel.purge()


def setup(bot):
   bot.add_cog(Main(bot))
