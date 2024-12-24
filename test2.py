import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.stats.anova import AnovaRM

def read_and_prepare_data():
    """
    実験データを読み込み、3要因分散分析用に整形する
    """
    # 実験の提示順序
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

    def get_condition_factors(condition_num):
        """条件番号から各要因の水準を返す"""
        # 条件1-4: 多い, 条件5-8: 少ない
        direction = 'many' if condition_num <= 4 else 'few'
        # 条件1,2,5,6: 早い, 条件3,4,7,8: 遅い
        speed = 'fast' if condition_num in [1,2,5,6] else 'slow'
        # 条件1,3,5,7: 止まる, 条件2,4,6,8: 止まらない
        stopping = 'stop' if condition_num in [1,3,5,7] else 'no_stop'
        return direction, speed, stopping

    data_list = []
    
    for exp_num in range(1, 9):
        try:
            df = pd.read_csv(f'data/本実験{exp_num}の解答フォーム.csv')
            lost_cols = [col for col in df.columns if '迷子になっていると感じた' in col]
            
            if not lost_cols:
                print(f"実験{exp_num}のデータに迷子感の評価列が見つかりません")
                continue
                
            presentation_order = condition_orders[exp_num]
            
            for subject_idx, row in df.iterrows():
                # 各被験者の各条件でのデータを処理
                for col_idx, col in enumerate(lost_cols):
                    if col_idx >= len(presentation_order):
                        continue
                    
                    # 提示順序から実際の条件番号を取得
                    actual_condition = presentation_order[col_idx]
                    # 条件番号から各要因の水準を取得
                    direction, speed, stopping = get_condition_factors(actual_condition)
                    
                    data_list.append({
                        'subject': f'subject_{exp_num}_{subject_idx}',
                        'actual_condition': actual_condition,
                        'presentation_order': col_idx + 1,
                        'direction': direction,
                        'speed': speed,
                        'stopping': stopping,
                        'score': row[col]
                    })
            
            print(f"実験{exp_num}のデータを読み込みました。")
            
        except Exception as e:
            print(f"実験{exp_num}の読み込み中にエラーが発生しました: {e}")
            continue
    
    df = pd.DataFrame(data_list)
    
    # データの整合性チェック
    print("\nデータ確認:")
    print(f"被験者数: {df['subject'].nunique()}")
    print("\n条件ごとのデータ数:")
    print(df.groupby(['direction', 'speed', 'stopping']).size().unstack(fill_value=0))
    
    # 各被験者が全条件を体験しているか確認
    conditions_per_subject = df.groupby('subject').size()
    if not all(conditions_per_subject == 8):
        print("\n警告: 一部の被験者のデータが8条件揃っていません")
        print(conditions_per_subject[conditions_per_subject != 8])
    
    return df

def analyze_three_way_interaction(df_3way):
    """3要因の繰り返し測定ANOVAを実行"""
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
    """条件ごとの平均値と標準偏差を計算"""
    means = df.groupby(['direction', 'speed', 'stopping'])['score'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    print("\n=== 条件ごとの記述統計量 ===")
    print(means)
    print("=" * 80)
    return means

def main():
    try:
        print("データを読み込んでいます...")
        df_3way = read_and_prepare_data()
        
        print("\n記述統計量を計算中...")
        means = calculate_condition_means(df_3way)
        
        print("\n3要因分散分析を実行中...")
        anova_result = analyze_three_way_interaction(df_3way)
        
    except Exception as e:
        import traceback
        print(f"エラーが発生しました: {e}")
        print("\nトレースバック:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()