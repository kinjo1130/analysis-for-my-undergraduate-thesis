import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.anova import AnovaRM

# 日本語フォントの設定（macOSの一般的なフォント）
plt.rcParams['font.family'] = 'Hiragino Sans GB'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB', 'Arial', 'Hiragino Kaku Gothic Pro', 'MS Gothic']
plt.rcParams['axes.unicode_minus'] = False

def read_experiment_data(exp_number):
    """
    各実験のCSVファイルを読み込み、データを整形する
    """
    # 実験条件の順序マッピング
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

    try:
        df = pd.read_csv(f'data/本実験{exp_number}の解答フォーム.csv')
        
        # 関連する列を抽出
        lost_cols = [col for col in df.columns if '迷子になっていると感じた' in col]
        confidence_cols = [col for col in df.columns if '自信を持って行動している' in col]
        looking_cols = [col for col in df.columns if '周囲を見回す回数が多い' in col]
        speed_cols = [col for col in df.columns if '歩行速度が早い' in col]
        stop_cols = [col for col in df.columns if '歩行中に立ち止まることが多い' in col]
        direction_cols = [col for col in df.columns if '進行方向を何度も変えている' in col]
        
        data_list = []
        for subject_idx, row in df.iterrows():
            for i in range(len(lost_cols)):
                if i >= len(condition_orders[exp_number]):
                    continue
                    
                condition = condition_orders[exp_number][i]
                # 条件の設定
                direction = '多い' if condition <= 4 else '少ない'
                speed = '早い' if condition in [1,2,5,6] else '遅い'
                stopping = '停止あり' if condition in [1,3,5,7] else '停止なし'
                
                data_list.append({
                    'subject': f'subject_{exp_number}_{subject_idx}',
                    'direction': direction,
                    'speed': speed,
                    'stopping': stopping,
                    'lost_feeling': row[lost_cols[i]],
                    'confidence': row[confidence_cols[i]],
                    'looking_around': row[looking_cols[i]],
                    'walking_speed': row[speed_cols[i]],
                    'stopping_freq': row[stop_cols[i]],
                    'direction_changes': row[direction_cols[i]]
                })
        
        return pd.DataFrame(data_list)
    
    except Exception as e:
        print(f"実験{exp_number}の読み込み中にエラーが発生しました: {e}")
        return None

def analyze_manipulation_check(data):
    """
    操作チェックの分析を実行
    """
    checks = {
        'A1_looking': {
            'variable': 'looking_around',
            'condition': 'direction',
            'high': '多い',
            'low': '少ない',
            'label': '周囲確認（要因A確認1）'
        },
        'A2_direction': {
            'variable': 'direction_changes',
            'condition': 'direction',
            'high': '多い',
            'low': '少ない',
            'label': '進行方向変更（要因A確認2）'
        },
        'B_speed': {
            'variable': 'walking_speed',
            'condition': 'speed',
            'high': '早い',
            'low': '遅い',
            'label': '歩行速度（要因B確認）'
        },
        'C_stopping': {
            'variable': 'stopping_freq',
            'condition': 'stopping',
            'high': '停止あり',
            'low': '停止なし',
            'label': '立ち止まり（要因C確認）'
        }
    }
    
    results = {}
    for key, check in checks.items():
        high_group = data[data[check['condition']] == check['high']][check['variable']]
        low_group = data[data[check['condition']] == check['low']][check['variable']]
        
        t_stat, p_val = stats.ttest_ind(high_group, low_group)
        
        results[key] = {
            'label': check['label'],
            'high_mean': high_group.mean(),
            'high_std': high_group.std(),
            'low_mean': low_group.mean(),
            'low_std': low_group.std(),
            't_stat': t_stat,
            'p_value': p_val
        }
    
    return results

def plot_manipulation_checks(data, results):
    """
    操作チェックの結果を可視化
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    axes = [ax1, ax2, ax3, ax4]
    
    for (key, result), ax in zip(results.items(), axes):
        conditions = ['高群', '低群']
        means = [result['high_mean'], result['low_mean']]
        sems = [result['high_std']/np.sqrt(len(data)/2), 
               result['low_std']/np.sqrt(len(data)/2)]
        
        ax.bar(conditions, means, yerr=sems, capsize=5)
        ax.set_title(result['label'])
        ax.set_ylabel('評価スコア')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 有意差マーカーの追加
        if result['p_value'] < 0.001:
            sig_text = '***'
        elif result['p_value'] < 0.01:
            sig_text = '**'
        elif result['p_value'] < 0.05:
            sig_text = '*'
        else:
            sig_text = 'n.s.'
        
        y_max = max(means) + max(sems) + 5
        ax.plot([0, 1], [y_max, y_max], 'k-')
        ax.text(0.5, y_max + 1, sig_text, ha='center')
    
    plt.tight_layout()
    plt.savefig('manipulation_checks.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_dependent_variables(data):
    """
    従属変数（迷子感と自信度）の分析
    """
    # 3要因分散分析（迷子感）
    anova_lost = AnovaRM(data, 'lost_feeling', 'subject',
                        within=['direction', 'speed', 'stopping']).fit()
    
    # 3要因分散分析（自信度）
    anova_confidence = AnovaRM(data, 'confidence', 'subject',
                             within=['direction', 'speed', 'stopping']).fit()
    
    return anova_lost, anova_confidence

def plot_dv_by_condition(data):
    """
    条件ごとの従属変数の平均値をプロット
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 迷子感の条件別平均
    means_lost = data.groupby(['direction', 'speed', 'stopping'])['lost_feeling'].mean().unstack()
    means_lost.plot(ax=ax1)
    ax1.set_title('条件別の迷子感スコア')
    ax1.set_xlabel('進行方向の変化')
    ax1.set_ylabel('迷子感スコア')
    ax1.grid(True)
    
    # 自信度の条件別平均
    means_conf = data.groupby(['direction', 'speed', 'stopping'])['confidence'].mean().unstack()
    means_conf.plot(ax=ax2)
    ax2.set_title('条件別の自信度スコア')
    ax2.set_xlabel('進行方向の変化')
    ax2.set_ylabel('自信度スコア')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('dv_by_condition.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # データの読み込みと結合
    all_data = []
    for exp_num in range(1, 9):
        exp_data = read_experiment_data(exp_num)
        if exp_data is not None:
            all_data.append(exp_data)
    
    if not all_data:
        raise ValueError("データが読み込めませんでした")
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 1. 操作チェックの分析
    print("\n=== 操作チェックの分析 ===")
    manip_results = analyze_manipulation_check(combined_data)
    for key, result in manip_results.items():
        print(f"\n{result['label']}:")
        print(f"高群平均: {result['high_mean']:.2f} (SD: {result['high_std']:.2f})")
        print(f"低群平均: {result['low_mean']:.2f} (SD: {result['low_std']:.2f})")
        print(f"t値: {result['t_stat']:.2f}, p値: {result['p_value']:.4f}")
    
    # 操作チェックの可視化
    plot_manipulation_checks(combined_data, manip_results)
    
    # 2. 従属変数の分析
    print("\n=== 従属変数の分析 ===")
    anova_lost, anova_confidence = analyze_dependent_variables(combined_data)
    
    print("\n迷子感の分散分析結果:")
    print("=" * 80)
    print(anova_lost)
    
    print("\n自信度の分散分析結果:")
    print("=" * 80)
    print(anova_confidence)
    
    # 従属変数の可視化
    plot_dv_by_condition(combined_data)
    
    print("\n分析結果を'manipulation_checks.png'と'dv_by_condition.png'として保存しました")

if __name__ == "__main__":
    main()