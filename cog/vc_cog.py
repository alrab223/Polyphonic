import asyncio
import codecs
import datetime
import json
import os
import re

import alkana
import discord
import torch
from discord.commands import Option
from discord.ext import commands
from scipy.io.wavfile import write

from vits import commons, utils
from vits.models import SynthesizerTrn
from vits.text import text_to_sequence


class Tts:

   def replace_english_kana(self, text):
      temp_text = text
      output = ""
      while word := re.search(r'[a-zA-Z]{4,}', temp_text):
         output += temp_text[:word.start()]
         if kana := alkana.get_kana(word.group().lower()):
            output += kana
         else:
            output += word.group()
         temp_text = temp_text[word.end():]
      output += temp_text
      return output

   def get_text(self, text, hps):
      text_norm = text_to_sequence(text, hps.data.text_cleaners)
      if hps.data.add_blank:
         text_norm = commons.intersperse(text_norm, 0)
      text_norm = torch.LongTensor(text_norm)
      return text_norm

   def read_censorship(self, text):
      pattern = "https?://[\\w/:%#\\$&\\?\\(\\)~\\.=\\+\\-]+"
      text = re.sub(pattern, "URL省略", text)
      if text.startswith("<") or text.startswith("!"):
         return ""
      elif text.count(os.linesep) > 4:
         return "改行が多数検出されたため、省略します"
      elif len(text) > 100:
         return "文字数が多いか、怪文書が検出されましたので省略します"
      else:
         text = text.replace("\n", "")
         text = self.replace_english_kana(text)  # 英語をカタカナに変換
         return text

   def tts_setting(self, message, length_scale=1.0):
      user_id = message.author.id
      text = self.read_censorship(message.content)  # 読み上げる一部文字列の変換
      if text != "":  # 文字が入っていれば読み上げ処理
         with open("json/user.json")as f:
            model = json.load(f)[str(user_id)]
      else:
         return
      config_path = f"vits/model/{model}/config.json"
      model_path = f"vits/model/{model}/model.pth"
      hps = utils.get_hparams_from_file(config_path)
      net_g = SynthesizerTrn(len(hps.symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).cuda()
      net_g.eval()
      utils.load_checkpoint(model_path, net_g, None)
      stn_tst = self.get_text(text, hps)
      with torch.no_grad():
         x_tst = stn_tst.cuda().unsqueeze(0)
         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
         audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.6,
                             noise_scale_w=0.668, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
      return audio, hps

   def tts_setting_free(self, text, model):
      text = self.replace_english_kana(text)  # 英語をカタカナに変換
      config_path = f"vits/model/{model}/config.json"
      model_path = f"vits/model/{model}/model.pth"
      hps = utils.get_hparams_from_file(config_path)
      net_g = SynthesizerTrn(len(hps.symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).cuda()
      net_g.eval()
      utils.load_checkpoint(model_path, net_g, None)
      stn_tst = self.get_text(text, hps)
      with torch.no_grad():
         x_tst = stn_tst.cuda().unsqueeze(0)
         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
         audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=0.6,
                             noise_scale_w=0.668, length_scale=1.0)[0][0, 0].data.cpu().float().numpy()
      return audio, hps

   async def voice_read(self, message):
      while self.voich.is_playing() is True:
         await asyncio.sleep(0.5)

      def delete_wav(c):  # 読み上げ後に削除
         os.remove(wav_path)
      date = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
      wav_path = f'wav/{date}.wav'
      audio, hps = self.tts_setting(message)
      write(wav_path, rate=hps.data.sampling_rate, data=audio)
      self.voich.play(discord.FFmpegPCMAudio(wav_path), after=delete_wav)
      self.voich.source = discord.PCMVolumeTransformer(self.voich.source)
      self.voich.source.volume = 0.5


class VcCommand(commands.Cog, Tts):

   def __init__(self, bot):
      self.bot = bot
      self.volume = 0.3
      self.voich = None

   def get_model_name(self):
      model_path = "vits/model"
      model_name = []
      files_dir = [x for x in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, x))]
      for i in files_dir:
         model_name.append(i)
      return model_name

   @commands.slash_command(name="botをボイスチャンネルに召喚")
   async def voice_connect(self, ctx):
      """botをボイチャに召喚します"""
      self.voich = await discord.VoiceChannel.connect(ctx.author.voice.channel)

   @commands.slash_command(name="読み上げ中止")
   async def stop(self, ctx):
      """読み上げを中止します"""
      if self.voich.is_playing():
         self.voich.stop()
         await ctx.respond("読み上げを中止しました", delete_after=3)
      else:
         await ctx.respond("読み上げ中ではありません", delete_after=3)

   @commands.slash_command(name="音声合成テスト")
   async def voice_generetor(self, ctx, model: Option(str, "モデルを選択", autocomplete=get_model_name), text: str):
      """音声合成を行います"""
      await ctx.respond("音声合成中です", delete_after=3)
      audio, hps = self.tts_setting_free(text, model)
      write("wav/temp.wav", rate=hps.data.sampling_rate, data=audio)
      if self.voich.is_playing():
         self.voich.stop()

      def delete_wav(c):  # 読み上げ後に削除
         os.remove("wav/temp.wav")
      self.voich.play(discord.FFmpegPCMAudio("wav/temp.wav"), after=delete_wav)
      self.voich.source = discord.PCMVolumeTransformer(self.voich.source)
      self.voich.source.volume = 0.5

   @commands.slash_command(name="読み上げモデル変更")
   async def model_change(self, ctx, model: Option(str, "モデルを選択", autocomplete=get_model_name)):
      # ユーザーデータの読み込み
      with open("json/user.json") as f:
         user_data = json.load(f)
      user_data[str(ctx.author.id)]= model
      # ユーザーデータの書き込み
      with open("json/user.json", "w") as f:
         json.dump(user_data, f, indent=3)

      await ctx.respond(f"モデルを{model}に変更しました")

   @commands.Cog.listener()
   async def on_ready(self):
      print("ログインしました")
      self.vc_ch = int(os.getenv("VC_CH"))
      self.read_ch = int(os.getenv("READ_TEXT_CH"))
      vc_channel = self.bot.get_channel(self.vc_ch)
      self.voich = await discord.VoiceChannel.connect(vc_channel)

   @commands.Cog.listener()
   async def on_message(self, message):
      # 条件を満たした書き込みを読み上げる
      if message.author.bot is False and self.voich is not None and message.channel.id == self.read_ch:
         await self.voice_read(message)


def setup(bot):
   bot.add_cog(VcCommand(bot))
