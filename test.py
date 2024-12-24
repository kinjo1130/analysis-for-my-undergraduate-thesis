import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.stats.anova import AnovaRM

def read_and_prepare_data():
    """
    実験データを読み込み、3要因分散分析用に整形する
    """
    # 各実験の条件順序
    condition_orders = {
        1: [1, 2, 8, 3, 7, 4, 6, 5],
        2: [2, 3, 1, 4, 8, 5, 7, 6],
        3: [3, 4, 2, 5, 1, 6, 8, 7],
        4: [4, 5, 3, 6, 2, 7, 1, 8],
        5: [5, 6, 4, 7, 3, 8, 2, 1],
        6: [6, 7, 5, 8, 4, 1, 3, 2],
        7: [7, 8, 6, 1, 5, 2, 4, 3],
        8: [8, 1, 7, 2, 6, 3, 5, 4]
    }

    data_list = []
    for exp_num in range(1, 9):
        try:
            df = pd.read_csv(f'data/本実験{exp_num}の解答フォーム.csv')
            lost_cols = [col for col in df.columns if '迷子になっていると感じた' in col]
            
            for subject_idx, row in df.iterrows():
                for i, col in enumerate(lost_cols):
                    if i >= len(condition_orders[exp_num]):
                        continue
                        
                    true_condition = condition_orders[exp_num][i]
                    
                    # 条件の設定
                    data_list.append({
                        'subject': f'subject_{exp_num}_{subject_idx}',
                        'direction': 'many' if true_condition <= 4 else 'few',  # 進行方向の変化
                        'speed': 'fast' if true_condition in [1,2,5,6] else 'slow',  # 歩行速度
                        'stopping': 'stop' if true_condition in [1,3,5,7] else 'no_stop',  # 立ち止まり
                        'score': row[col]  # 迷子感スコア
                    })
            
        except Exception as e:
            print(f"実験{exp_num}の読み込み中にエラーが発生しました: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    
    # データの整合性チェック
    print("\nデータ確認:")
    print(f"被験者数: {df['subject'].nunique()}")
    print(f"各条件の組み合わせ数:")
    print(df.groupby(['direction', 'speed', 'stopping']).size())
    
    return df

def analyze_three_way_interaction(df_3way):
    """
    3要因の繰り返し測定ANOVAを実行
    """
    model = AnovaRM(data=df_3way,
                    depvar='score',
                    subject='subject',
                    within=['direction', 'speed', 'stopping'])
    
    anova_result = model.fit()
    
    print("\n=== 3要因繰り返し測定ANOVA 結果 ===")
    print(anova_result)
    print("=" * 80)
    
    return anova_result

def calculate_condition_means(df):
    """
    条件ごとの平均値と標準偏差を計算
    """
    means = df.groupby(['direction', 'speed', 'stopping'])['score'].agg(['mean', 'std', 'count']).round(2)
    print("\n=== 条件ごとの記述統計量 ===")
    print(means)
    print("=" * 80)
    return means

def main():
    try:
        # データの読み込みと整形
        print("データを読み込んでいます...")
        df_3way = read_and_prepare_data()
        
        # 記述統計量の計算
        means = calculate_condition_means(df_3way)
        
        # 3要因分散分析の実行
        print("\n3要因分散分析を実行中...")
        anova_result = analyze_three_way_interaction(df_3way)
        
    except Exception as e:
        import traceback
        print(f"エラーが発生しました: {e}")
        print("\nトレースバック:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()