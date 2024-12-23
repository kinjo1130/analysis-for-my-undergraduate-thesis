import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib

def analyze_multiple_experiments():
   dfs = []
   for i in range(1, 9):
       try:
           df = pd.read_csv(f'./data/本実験{i}の解答フォーム.csv')
           dfs.append(df)
       except FileNotFoundError:
           print(f"ファイル 本実験{i}の解答フォーム.csv が見つかりません")
           continue
   
   if not dfs:
       raise Exception("データファイルが読み込めませんでした")
   
   combined_df = pd.concat(dfs, ignore_index=True)
   
   behaviors = [
       '周囲を見回す回数が多い',
       '進行方向を何度も変えている',
       '歩行速度が早い',
       '歩行中に立ち止まることが多い'
   ]
   
   results = pd.DataFrame(columns=['行動特徴', 'F値', 'p値', '相関係数'])
   detailed_results = {}
   
   for behavior in behaviors:
       behavior_cols = [col for col in combined_df.columns if behavior in col]
       lost_cols = [col for col in combined_df.columns if '迷子になっていると感じた' in col]
       
       behavior_vals = combined_df[behavior_cols].values.flatten()
       lost_vals = combined_df[lost_cols].values.flatten()
       
       correlation = stats.pearsonr(behavior_vals, lost_vals)
       
       median = np.median(behavior_vals)
       high_group = lost_vals[behavior_vals >= median]
       low_group = lost_vals[behavior_vals < median]
       f_stat, p_val = stats.f_oneway(high_group, low_group)
       
       results.loc[len(results)] = [
           behavior,
           f_stat,
           p_val,
           correlation[0]
       ]
       
       detailed_results[behavior] = {
           'mean': np.mean(behavior_vals),
           'std': np.std(behavior_vals),
           'median': median,
           'correlation': correlation[0],
           'p_value': p_val,
           'f_stat': f_stat
       }
   
   return results, detailed_results

def display_results(results):
   print("\n迷子行動の分析結果:")
   print("=" * 70)
   print(f"{'行動特徴':^30} {'F値':^12} {'p値':^12} {'相関係数':^12}")
   print("-" * 70)
   
   for _, row in results.iterrows():
       behavior = row['行動特徴']
       f_val = f"{row['F値']:.3f}".rjust(12)
       p_val = f"{row['p値']:.3f}".rjust(12)
       corr = f"{row['相関係数']:.3f}".rjust(12)
       print(f"{behavior:<30} {f_val} {p_val} {corr}")
   
   print("=" * 70)
   print("\n* p < 0.05, ** p < 0.01")

def plot_correlation(results):
   plt.figure(figsize=(12, 6))
   colors = ['skyblue' if p >= 0.05 else 'lightgreen' if p >= 0.01 else 'lightcoral' 
             for p in results['p値']]
   
   bars = plt.bar(results['行動特徴'], results['相関係数'], color=colors)
   plt.xticks(rotation=45, ha='right')
   plt.title('各行動特徴と迷子判断の相関係数')
   plt.ylabel('相関係数')
   
   for i, p in enumerate(results['p値']):
       if p < 0.01:
           plt.text(i, results['相関係数'][i], '**', ha='center', va='bottom')
       elif p < 0.05:
           plt.text(i, results['相関係数'][i], '*', ha='center', va='bottom')
   
   plt.tight_layout()
   plt.grid(True, axis='y', linestyle='--', alpha=0.7)
   plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
   plt.close()

def display_detailed_results(detailed_results):
   print("\n詳細な統計結果:")
   print("=" * 100)
   print(f"{'行動特徴':^30} {'平均':^12} {'標準偏差':^12} {'中央値':^12} {'相関係数':^12} {'p値':^12}")
   print("-" * 100)
   
   for behavior, stats_dict in detailed_results.items():
       print(f"{behavior:<30} "
             f"{stats_dict['mean']:>12.3f} "
             f"{stats_dict['std']:>12.3f} "
             f"{stats_dict['median']:>12.3f} "
             f"{stats_dict['correlation']:>12.3f} "
             f"{stats_dict['p_value']:>12.3f}")
   print("=" * 100)

def main():
   try:
       results, detailed_results = analyze_multiple_experiments()
       display_results(results)
       display_detailed_results(detailed_results)
       plot_correlation(results)
       print("\n分析が完了しました。グラフは'correlation_plot.png'として保存されました。")
   except Exception as e:
       print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
   main()