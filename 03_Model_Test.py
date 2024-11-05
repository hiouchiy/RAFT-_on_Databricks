# Databricks notebook source
# MAGIC %md
# MAGIC ## 環境：サーバーレス・ノートブック

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install mlflow[databricks]
# MAGIC %pip install torch==2.3.1
# MAGIC %pip install -U transformers
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

SYSTEM_PROMPT = """* You are an excellent AI assistant.
* Please respond to user questions based on the reference information enclosed within  <DOCUMENT> and </DOCUMENT>. However, some of the reference information may be irrelevant, so please ignore any unrelated parts.
* Please provide your response in Japanese."""

question = "IBM 1401はどこにありましたか？"
# question = "RISDでの学生が求められていたものは何ですか？"
# question = "『ANSI Common Lisp』の表紙の絵はいつ描かれましたか？"
# question = "著者がLispに関する別の本を書こうと思った理由は何ですか？"
# question = "ニューヨークで著者がスタジオアシスタントを務めていた画家の名前は何ですか？"
# question = "インターリーフで働いている間に、著者は何に多くの時間を費やしていましたか？"
# question = "RISDでの学生の中で意気消沈していたのはどのような学生でしたか？"
# question = "著者がインターリーフを辞めた理由は何でしたか？"
prompt = f"""
<DOCUMENT>assistant: 私たちは小売業について、知りたかった以上のことを学びました。例えば、男性用シャツの小さな画像しか持てない場合（当時の基準ではすべての画像が小さかったのですが）、シャツ全体の写真よりも襟のクローズアップを持つ方が良いということです。これを学んだことを覚えている理由は、男性用シャツの画像を約30枚再スキャンしなければならなかったからです。最初のスキャンセットはとても美しかったのに。</DOCUMENT>
<DOCUMENT>assistant: そうなんです。これが、安価なソフトウェアが高価なソフトウェアを駆逐する傾向があることを学んだ方法でした。ただ、インターリーフにはまだ数年の寿命が残されていました。

インターリーフは大胆な手を打ちました。Emacsに触発され、スクリプト言語を追加することにし、その言語はLispの方言を採用しました。彼らはこれを使ってプログラムを組むために、Lispハッカーを募集していました。これは私が今まで経験した中で最も「まとも」な仕事に近いものでしたが、残念ながら私は理想的な従業員とは程遠く、上司や同僚に迷惑をかけました。彼らが採用したLispは巨大なCプログラムの表層に過ぎず、Cを知らない私には理解しがたいものでしたし、Cを学ぶ意欲もありませんでした。加えて、私は非常に無責任でした。当時、プログラミングの仕事は毎日定時に出勤するのが普通で、私にはそれがどうにも不自然に感じられました。今となっては、時代が私の考えに近づいてきているものの、当時は摩擦が生まれました。年末にかけて、『On Lisp』の執筆にこっそりと時間を割いていましたが、その時点で出版契約を結んでいました。

良かったことといえば、特に美術学生の水準からすると、かなりの収入が得られたことです。フィレンツェでは家賃を支払った後、1日の予算はわずか7ドル程度でしたが、今では会議に座っているだけで、その4倍以上の報酬が1時間に得られるようになっていました。質素な生活をすることで、RISDに戻るための資金を貯めることができ、大学のローンも返済しました。

インターリーフでの経験から学んだことは、ほとんどが「こうしない方が良い」というものでしたが、いくつか有益な教訓もありました。営業よりも製品開発が主導する技術企業が成功すること、しかし営業は本物のスキルであり、それに秀でた人は本当に優れていること、複数人でコードを編集するとバグが増えること、安いオフィスは時に鬱屈した気分にさせること、計画された会議は即興の会話に劣ること、大規模な官僚的な顧客は安定した資金源ではないこと、そして一般的な勤務時間は最も生産的なプログラミング時間とはほとんど重ならないことです。

しかし最も重要な教訓は、ViawebやY Combinatorの両方で活かされた、「安価なものが高価なものを駆逐する」という原則でした。つまり、「エントリーレベル」の製品であることは名声が少ないかもしれませんが、他の誰かがその地位を得てしまうと、あなたを越える存在になってしまうのです。要するに、名声は時に危険信号であるということです。

翌年の秋、RISDに戻るためにインターリーフを辞め、顧客向けプロジェクトのためにフリーランスで働けるよう手配しました。これがしばらくの間、私の収入源となりました。後にプロジェクトで訪れた際、ある人からHTMLという新しい技術について聞かされました。それはSGMLの派生物だと説明されましたが、インターリーフでの経験からマークアップ言語の話には警戒していたため、その時は聞き流しました。しかし、このHTMLがその後私の人生で大きな役割を果たすことになるのです。

1992年の秋、私はRISDでの学びを続けるためプロビデンスに戻りました。基礎教育はあくまで入門的なもので、アカデミアは文明的ながらもどこか皮肉な空間でした。今回は本物の美術学校がどんなものかを見届けるつもりでしたが、実際は期待とは少し違いました。絵画学科は想像していたより厳格ではなく、隣のテキスタイル学科のほうがよほど厳しそうでした。絵画学生は自己表現が求められ、より世俗的な学生は独自のシグネチャースタイルを確立することが奨励されていました。

シグネチャースタイルとは、ショービジネスでいう「シュティック」に相当するもので、その作品が誰のものかを瞬時に識別できる特徴です。例えば、ある特定の画風の絵を見れば、それがロイ・リキテンスタインのものだと分かります。このような作品が高価な理由の一つです。

真剣な学生も少なからずいました。高校時代から絵が得意だった彼らはRISDで更なる技術を学ぼうとしていましたが、その期待にRISDは必ずしも応えませんでした。それでも彼らは続けていました。私は高校で「絵が得意」な子どもではなかったため、こうした真剣な学生により近い存在だったかもしれません。

RISDで受けた色彩のクラスは有益でしたが、それ以外は基本的に自習で学んでおり、それは無料で行えるものでした。そのため1993年に退学し、プロビデンスにしばらく滞在しました。その後、友人のナンシー・パーメットの計らいで、ニューヨークにある家賃統制されたアパートに住めることになりました。そこは現在の住まいと大差ない家賃で、ニューヨークはアーティストが集う場所とされていました。私は迷わず引っ越しを決めました。

アステリックスの漫画では、ローマのガリアを支配していない一角がズームインされますが、同様に1993年当時のニューヨーク市でもアッパーイーストサイドの中に裕福ではない一角がありました。そこが「ヨークビル」と呼ばれる私の新しい住まいでした。今や私はニューヨークで絵を描く「ニューヨークのアーティスト」になったのです。

私はお金の問題を心配していました。インターリーフが下り坂になっていることを感じ取っていたからです。Lispでのフリーランスの仕事は稀で、他の言語でプログラムするのは嫌でした。そこで、Lispについての本をもう一冊書くことに決めました。教科書として使用されるような人気の本を目指しました。印税で生活し、時間を全て絵画に費やす生活を想像していました。（この頃に書いた本『ANSI Common Lisp』の表紙絵は私が描いたものです。）

ニューヨークで良かったのは、イデル・ウェーバーとその夫ジュリアンの存在でした。イデル・ウェーバーは初期のフォトリアリストの一人で、私はハーバードで彼女の絵画クラスを受けました。ニューヨークに移った後、私は彼女のスタジオアシスタントのような役割を担いました。

1994年末頃、ラジオで有名なファンドマネージャーの話を聞きました。彼は私より少し年上で非常に裕福でした。その時、なぜ自分も裕福にならないのかと思い、裕福になれれば好きなことに集中できるだろうと考えました。

ちょうどその頃、新たなもの「ワールドワイドウェブ」についての話が増えていました。ハーバードの友人ロバート・モリスがこれを見せてくれましたが、私はウェブが大きな変革をもたらすと感じました。コンピュータの普及にGUIが果たした役割と同様に、ウェブはインターネットに同じ影響を与えるだろうと予感しました。

裕福になるなら、次の機会が訪れたと感じていました。そしてその直感は正しかったのです。</DOCUMENT>
<DOCUMENT>assistant: 当時、初心者の作家が書くべきとされていたもの、そしておそらく今でもそうである短編小説を書きました。私の物語はひどいものでした。ほとんどプロットがなく、ただ強い感情を持ったキャラクターがいるだけで、それが深みを与えていると想像していました。

最初にプログラムを書こうとしたのは、当時「データ処理」と呼ばれていたことに使われていたIBM 1401でした。これは9年生の時で、私は13歳か14歳でした。学区の1401はたまたま私たちの中学校の地下にあり、友人のリッチ・ドレーブスと私はそれを使う許可を得ました。そこはまるでミニ・ボンド悪役の隠れ家のようで、CPU、ディスクドライブ、プリンター、カードリーダーといった異星人のような機械が明るい蛍光灯の下で高床に並んでいました。

私たちが使った言語は初期のFortranでした。プログラムをパンチカードにタイプし、それをカードリーダーに積んでボタンを押してプログラムをメモリにロードし、実行する必要がありました。結果は通常、非常にうるさいプリンターで何かを印刷することでした。

私は1401に困惑していました。どう扱えばいいのか分かりませんでした。</DOCUMENT>
{question}
"""

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファインチューニング前

# COMMAND ----------

import mlflow
from mlflow import MlflowClient  
import pandas as pd  
import numpy as np  
import torch
import gc
  
# 必要なライブラリを追加  
gc.collect()  
torch.cuda.empty_cache()

# カタログとテーマを設定  
catalog = "system"  
schema = "ai"
model_name = "meta_llama_3_8b_instruct"

# MLflowの設定  
mlflow.set_registry_uri("databricks-uc")  
mlflow_client = MlflowClient()  
  
# 最新バージョンのモデルを取得する関数  
def get_latest_model_version(mlflow_client, registered_name):  
    latest_version = 1  
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):  
        version_int = int(mv.version)  
        if version_int > latest_version:  
            latest_version = version_int  
    return latest_version  
  
# 登録されたモデルの名前を設定  
registered_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(mlflow_client, registered_name)  
logged_model = f"models:/{registered_name}/{model_version}"  

# モデルをロード  
loaded_model = mlflow.pyfunc.load_model(logged_model)

print(loaded_model)

# COMMAND ----------

# 辞書形式でプロンプトを定義  
data = {  
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt},
    ],
    "max_tokens": 500,
    "temperature": 0.1,
} 

# 推論を実行
response = loaded_model.predict(data)
# print(response)

# 'choices'の'text'を取得してprint
text_response = response[0]['choices'][0]['message']['content']
print(text_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ファインチューニング後

# COMMAND ----------

import mlflow
from mlflow import MlflowClient  
import pandas as pd  
import numpy as np  
import torch
import gc
  
# 必要なライブラリを追加  
gc.collect()  
torch.cuda.empty_cache()

# カタログとテーマを設定  
catalog = "hiroshi"  
schema = "raft"
model_name = "llama31_8b_ift"

# MLflowの設定  
mlflow.set_registry_uri("databricks-uc")  
mlflow_client = MlflowClient()  
  
# 最新バージョンのモデルを取得する関数  
def get_latest_model_version(mlflow_client, registered_name):  
    latest_version = 1  
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):  
        version_int = int(mv.version)  
        if version_int > latest_version:  
            latest_version = version_int  
    return latest_version  
  
# 登録されたモデルの名前を設定  
registered_name = f"{catalog}.{schema}.{model_name}"
model_version = get_latest_model_version(mlflow_client, registered_name)  
logged_model = f"models:/{registered_name}/{model_version}"  

# モデルをロード  
loaded_model = mlflow.pyfunc.load_model(logged_model)

print(loaded_model)

# COMMAND ----------

# 辞書形式でプロンプトを定義  
data = {  
    "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt},
    ],
    "max_tokens": 200,
    "temperature": 0.1,
} 

# 推論を実行
response = loaded_model.predict(data)
# print(response)

# 'choices'の'text'を取得してprint
text_response = response[0]['choices'][0]['message']['content']
print(text_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ### エンドポイントを呼び出す

# COMMAND ----------

import requests
import json

data = {
  "messages": [
      {"role": "system", "content": SYSTEM_PROMPT},
      {"role": "user", "content": prompt},
    ],
    "max_tokens": 500,
    "temperature": 0.1,
}

# Databricks URL とトークン
databricks_host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {"Context-Type": "text/json", "Authorization": f"Bearer {databricks_token}"}

response = requests.post(
    url=f"{databricks_host}/serving-endpoints/hiroshi-raft-llama31-8b/invocations", json=data, headers=headers
)

print(response.json()['choices'][0]['message']['content'])

# COMMAND ----------


