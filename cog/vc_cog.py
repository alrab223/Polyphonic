import asyncio
import datetime
import json
import os
import re

import alkana
import discord
import torch
from discord.commands import option
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

   def single_voice_make(self, model, text, length_scale=1.0):
      config_path = f"vits/single_model/{model}/config.json"
      model_path = f"vits/single_model/{model}/model.pth"
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

   def multi_voice_make(self, model, id, text, title, length_scale=1.0):
      config_path = f"vits/multi_model/{title}/config.json"
      model_path = f"vits/multi_model/{title}/model.pth"
      hps = utils.get_hparams_from_file(config_path)
      model = SynthesizerTrn(len(hps.symbols), hps.data.filter_length // 2 + 1, hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model).cuda()
      utils.load_checkpoint(model_path, model, None)
      net_g = model.eval()
      stn_tst = self.get_text(text, hps)
      sp_id = id
      with torch.no_grad():
         x_tst = stn_tst.cuda().unsqueeze(0)
         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
         sid = torch.LongTensor([sp_id]).cuda()
         audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.6,
                             noise_scale_w=0.668, length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
      return audio, hps

   def tts_setting(self, message, length_scale=1.0):
      user_id = message.author.id
      text = self.read_censorship(message.content)  # 読み上げる一部文字列の変換
      if text != "":  # 文字が入っていれば読み上げ処理
         with open("json/user.json")as f:
            dic = json.load(f)
            model = dic[str(user_id)]["speaker_name"]
            id = dic[str(user_id)]["id"]
            title = dic[str(user_id)]["title"]
      else:
         return
      # モデルがシングルかマルチかで分岐
      if id is None:
         return self.single_voice_make(model, text, length_scale)
      else:
         return self.multi_voice_make(model, id, text, title, length_scale)

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
      self.ba_flag = 1

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

   async def get_title(self, ctx: discord.AutocompleteContext):
      model_path = "vits/multi_model"
      title_list = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
      return [title for title in title_list if ctx.value.lower() in title.lower()]

   async def get_model_type(self, ctx: discord.AutocompleteContext):
      model_path = "vits/multi_model"
      title_list = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
      title_list.append("single_model")
      return [title for title in title_list if ctx.value.lower() in title.lower()]

   async def get_model(self, ctx: discord.AutocompleteContext):
      if ctx.options["model_type"] == "single_model":
         model_path = "vits/single_model"
         model_list = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
         return [model for model in model_list if ctx.value in model]
      else:
         with open(f"vits/multi_model/{ctx.options['model_type']}/chara.json", encoding="utf-8") as f:
            chara = json.load(f)
            chara_list = [k for k, v in chara.items()]
         return [chara for chara in chara_list if ctx.value in chara]

   async def get_single_model(self, ctx: discord.AutocompleteContext):
      model_path = "vits/single_model"
      model_list = [f for f in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, f))]
      return [model for model in model_list if ctx.value in model]

   async def get_multi_model(self, ctx: discord.AutocompleteContext):
      with open(f"vits/multi_model/{ctx.options['title']}/chara.json", encoding="utf-8") as f:
         chara = json.load(f)
      chara_list = [k for k, v in chara.items()]

      return [chara for chara in chara_list if ctx.value in chara]

   def json_edit(self, ctx, title, speaker_name):
      with open("json/user.json") as f:
         user_data = json.load(f)

      # シングルモデルかマルチモデルかで分岐
      if title == "single_model":
         user_data[str(ctx.author.id)]["id"] = None
         user_data[str(ctx.author.id)]["title"] = None
         user_data[str(ctx.author.id)]["speaker_name"] = speaker_name
      else:
         with open(f"vits/multi_model/{title}/chara.json", encoding="utf-8") as f:
            dic = json.load(f)
         user_data[str(ctx.author.id)]["id"] = dic[speaker_name]
         user_data[str(ctx.author.id)]["title"] = title
         user_data[str(ctx.author.id)]["speaker_name"] = speaker_name

      # ユーザーデータの書き込み
      with open("json/user.json", "w") as f:
         json.dump(user_data, f, indent=3)
      return speaker_name

   @commands.slash_command(name="読み上げモデル変更")
   @option("model_type", description="モデルの種類を選択", autocomplete=get_model_type)
   @option("speaker_name", description="モデルを選択", autocomplete=get_model)
   async def multi_model_change(self, ctx, model_type: str, speaker_name: str):
      self.json_edit(ctx, model_type, speaker_name)
      await ctx.respond(f"モデルを{speaker_name}に変更しました")

   @commands.slash_command(name="朗読")
   @option("model_type", description="モデルの種類を選択", autocomplete=get_model_type)
   @option("speaker_name", description="モデルを選択", autocomplete=get_model)
   async def recitation(self, ctx, model_type, speaker_name, text: str, length_scale: float = 1.0):
      """朗読を行います"""
      await ctx.respond("音声合成中です", delete_after=3)
      
      if model_type == "single_model":
         audio, hps = self.single_voice_make(speaker_name, text, length_scale)
      else:
         with open(f"vits/multi_model/{model_type}/chara.json", encoding="utf-8") as f:
            dic = json.load(f)
         speaker_id = dic[speaker_name]
         audio, hps = self.multi_voice_make(speaker_name, speaker_id, text, model_type, length_scale)
      write("wav/temp.wav", rate=hps.data.sampling_rate, data=audio)

      if self.voich.is_playing():
         self.voich.stop()

      def delete_wav(c):  # 読み上げ後に削除
         os.remove("wav/temp.wav")
      self.voich.play(discord.FFmpegPCMAudio("wav/temp.wav"), after=delete_wav)
      self.voich.source = discord.PCMVolumeTransformer(self.voich.source)
      self.voich.source.volume = 0.5

   @commands.Cog.listener()
   async def on_ready(self):
      print("ログインしました")
      self.vc_ch = int(os.getenv("VC_CH"))
      self.read_ch = int(os.getenv("READ_TEXT_CH"))
      vc_channel = self.bot.get_channel(self.vc_ch)
      self.voich = await discord.VoiceChannel.connect(vc_channel)

   @commands.Cog.listener()
   async def on_message(self, message):
      if message.author.bot is False and self.voich is not None and message.channel.id == self.read_ch:
         await self.voice_read(message)


def setup(bot):
   bot.add_cog(VcCommand(bot))
