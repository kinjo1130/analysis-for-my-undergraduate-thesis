# 卒論の実験の分析プログラム(分散分析)

## 環境構築

1. Python仮想環境のセットアップ
```bash
# Python3とvenvのインストール確認
python3 --version

# 仮想環境の作成
python3 -m venv venv

# 仮想環境の有効化
source venv/bin/activate  # Mac/Linux
# または
venv\Scripts\activate  # Windows

pip install --upgrade pip

# PyTorchのインストール（オプション）
pip install torch torchvision torchaudio

# 分析に必要なパッケージ
pip install pandas numpy scipy matplotlib seaborn statsmodels japanize-matplotlib

python -c "import torch; print(torch.__version__)"


## プロジェクト構成
analyze/
├── data/
│   ├── 本実験1の解答フォーム.csv
│   ├── 本実験2の解答フォーム.csv
│   └── ...
├── venv/
└── analyze.py

## 実行方法
`python index.py`
