* エピソード数で学習バッチを指定できるようにした（もともとのコードは、時間ステップ数で指定）。
* traj_percentileを調整することにより、各エピソードの総報酬の低いものからX%のエピソードだけを取り出して、学習に用いることができるようにした。これは、conditional value at riskの最適化指標を目的関数としていることに相当する。
* bashファイルを実行すると、(1)100回の反復による訓練データで各バッチが32エピソードの場合（100percentileの軌道で、計3200エピソード）、(2)合計1e6ステップの訓練データで各バッチが5000ステップの場合（結果的にたまたま3000エピソード弱）、

