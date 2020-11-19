![comp](./data/info/images/readme/001_comp.png)
# kaggle-Rock-Paper-Scissors
Rock, Paper, Scissorsコンペのリポジトリ

- directory tree
```
Kaggle-Cornell-Birdcall-Identification
├── README.md
└── data
    └── info
        └── images
            └── readme  <---- readmeで使用するイメージ
```

## Paper
|No.|Status|Name|Detail|Date|Url|
|---|---|---|---|---|---|
|01|<font color='gray'>Done</font>|音響イベントと音響シーンの分析|日本語記事。まず最初に読むとよい。|2018|[url](https://www.jstage.jst.go.jp/article/jasj/74/4/74_198/_pdf)|

## Basics

**Description(DeepL)**

ロック、ペーパー、シザー（時々 roshambo と呼ばれる）は、遊び場の意見の相違を解決したり、誰が道路の旅で前の座席に乗るために取得を決定するための定番となっています。ゲームはシンプルで、力のバランスが取れています。3つの選択肢があり、それぞれが他の2人に勝つか負けるかを選択できます。一連の本当にランダムなゲームでは、それぞれのプレイヤーが勝ったり負けたり、ゲームの大体3分の1を引き分けたりします。しかし、人間は本当にランダムではないので、AIに楽しい機会を与えてくれます。

研究では、ロック、ペーパー、シザーAIが一貫して人間の対戦相手を打ち負かすことができることが示されています。過去のゲームをインプットとして、それはパターンを研究してプレイヤーの傾向を理解する。しかし、単純な「ベスト・オブ・3」のゲームを「ベスト・オブ・1000」に拡張するとどうなるのだろうか。人工知能はどれだけのパフォーマンスを発揮できるのでしょうか？

このシミュレーション大会では、この古典的なゲームを何度もラウンドを重ねて対戦するAIを作成します。あなたのAIが負けるよりも、あなたのAIが勝つパターンを見つけることができるでしょうか？マッチがランダムではないエージェントを含む場合、ランダムなプレイヤーを大幅に上回ることが可能です。強いAIは、予測可能なAIに一貫して勝つことができる。

この問題は、機械学習、人工知能、データ圧縮の分野では基本的な問題です。人間の心理学や階層的時間記憶への応用の可能性さえある。手を温めて、この課題でRock, Paper, Scissorsの準備をしましょう。

画像の謝辞。
写真はThe Noun Projectより。Rock, Paper, Scissors

**Evaluation(DeepL)**

毎日、あなたのチームは最大5人のエージェント（ボット）を大会に提出することができます。各エージェントは、同じようなスキル評価を持つ他のボットとのエピソード（ゲーム）で対戦します。時間が経つにつれて、スキルの評価は、勝てば上がる、負ければ下がるようになります。提出されたボットは、大会終了までゲームをプレイし続けます。リーダーボードには、最高得点を獲得したボットのみが表示されますが、提出物ページではすべての提出物の進捗状況を確認することができます。

各提出物には推定スキル評価があり、ガウシアンN(μ,σ2)でモデル化されており、μは推定スキル、σは推定スキルの不確実性を表し、時間の経過とともに減少します。

提出物をアップロードすると、最初に検証エピソードを再生し、提出物が正しく動作するかどうかを確認します。エピソードが失敗した場合、提出物はエラーとしてマークされます。そうでない場合は、サブミッションをμ0=600で初期化し、継続的な評価のためにすべてのサブミッションのプールに参加します。

すべての提出物のプールからエピソードを繰り返し実行し、公正に一致するように類似の評価を持つ提出物を選択します。1日に8エピソードを実行することを目標としていますが、より早くフィードバックを得られるように、最新の投稿エピソードには若干の追加料金を加えています。

エピソードが終了すると、そのエピソードに含まれるすべての提出物の評価が更新されます。あるサブミッションが勝った場合は、そのサブミッションのμ値を増加させ、相手のμ値を減少させます。更新は、以前のμ値に基づく期待される結果からの偏差と、各提出物の不確実性σとの相対的な大きさを持つことになります。また、結果によって得られる情報量に応じてσの項を減らします。あなたのボットがエピソードに勝ったり負けたりするスコアは、スキル評価の更新には影響しません。

投稿締め切り時に、追加投稿はロックされます。エピソードを継続して実行するために、さらに1週間が割り当てられます。この週の終了時に、リーダーボードは最終的なものとなります。

## Log
### 20201115
- 参加開始！
- kaggle日記の作成 ([参考](https://github.com/fkubota/kaggle-Cornell-Birdcall-Identification))
- 多腕バンディットが現状ベストスコア(633.2)

### 20201117
- スコアリングの変更点
	- スコアが20以上勝ち越していないと勝利判定されない
	- 50％で高いレーティングのbotと対戦できるように

### 20201119
- 多腕バンディットアルゴリズムの実装コード解読

```Python
# 全エージェントのベースとなるクラス
class agent():
	def initial_step(self):
		return np.random.randint(3)

	def history_step(self, history):
		return np.random.randint(3)

	def step(self, history):
		if len(history) == 0:
			return self.initial_step()
		else:
			return self.history_step(history)

# エージェント例：前の対戦相手の手＋シフトを返すエージェント
class mirror_shift(agent):
	def __init__(self, shift=0):
		self.shift = shift

	def history_step(self, history):
		return (history[-1]['competitorStep'] + self.shift) % 3

# history.csvを読み込む関数
def load_history(file = "history.csv"):
	return pd.read_csv(file).to_dict('records')

# 対戦相手の最新手をhistoryに追記する関数
def update_competitor_step(history, competitorStep):
	history[-1]['competitorStep'] = competitorStep
	return history

# 手を記録する関数
def log_step(step = None, history = None, agent = None, competitorStep = None):
	if step is None:
		step = np.random.randint(3)
	if history is None:
		history = []
	history.append({'step' : step, 'competitorStep' : competitorStep, 'agent' : agent})
	save_hisory(history)
	return step

if observation.step == 0:
	history = []
	bandit_state = {k:[1, 1] for k in agents.keys()}
else:
	history = update_competitor_step(load_history(), observation.lastOpponentAction)

	# バンディットの状態を読み込み
	with open('bandit.json') as json_file:
		bandit_state = json.load(json_file)

	# 前回の対戦結果を元にバンディットの状態を更新
	if (history[-1]['competitorStep'] - history[-1]['step']) % 3 == 1:
		bandit_state[history[-1]['agent']][1] += 1
	elif (history[-1]['competitorStep'] - history[-1]['step']) % 3 == 2:
		bandit_state[history[-1]['agent']][0] += 1
	else:
		bandit_state[history[-1]['agent']][0] += 0.5
		bandit_state[history[-1]['agent']][1] += 0.5
with open('bandit.json', 'w') as outfile:
	json.dump(bandit_state, outfile)

# 使用するエージェントを決定するためにベータ分布から乱数を発生
best_proba = -1
best_agent = None
for k in bandit_state.keys():
	proba = np.random.beta(bandit_state[k][0], bandit_state[k][1])
	if proba > best_proba:
		best_proba = proba
		best_agent = k

step = agents[best_agent].step(history)
return log_step(step, history, best_agent)


```
