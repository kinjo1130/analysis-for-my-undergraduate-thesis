import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
from statsmodels.stats.anova import AnovaRM

def analyze_three_way_interaction(df_3way):
    """
    3要因(A,B,C)の繰り返し測定ANOVAを行い、交互作用を含む結果を表示する。
    
    事前に df_3way の中に以下の列があることを想定:
        - 'subject': 被験者ID (同一被験者は同じ値)
        - 'A': 要因A (例: '多い'/'少ない')
        - 'B': 要因B (例: '速い'/'遅い')
        - 'C': 要因C (例: '止まる'/'止まらない')
        - 'score': 迷子感 or 他の測定値

    Returns
    -------
    anova_result : AnovaRM fit 結果オブジェクト
    """
    model = AnovaRM(data=df_3way,
                    depvar='score',
                    subject='subject',
                    within=['A','B','C'])  # 3要因
    anova_result = model.fit()
    
    print("\n--- 3要因繰り返し測定ANOVA 結果 ---")
    print(anova_result)
    print("------------------------------------\n")
    return anova_result



def label_significance(p_value):
    """
    p値に応じて有意差ラベルを返す:
      - **  (p < 0.01)
      - *   (p < 0.05)
      - ''  (それ以外)
    """
    if p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


def analyze_multiple_experiments():
    """
    複数の実験CSVファイルを読み込み、行動特徴と『迷子に感じる度合い』との
    ピアソン相関・分散分析を行う。
    
    Returns
    -------
    results : pd.DataFrame
        行動特徴ごとのF値・p値・相関係数・相関p値をまとめたテーブル
    detailed_results : dict
        行動特徴ごとの詳細統計（平均・標準偏差・中央値・相関・相関p値・F値・p値 など）
    """
    dfs = []
    for i in range(1, 9):
        filepath = f'./data/本実験{i}の解答フォーム.csv'
        try:
            df = pd.read_csv(filepath)
            dfs.append(df)
        except FileNotFoundError:
            print(f"ファイル {filepath} が見つかりません。スキップします。")
    
    if not dfs:
        raise FileNotFoundError("いずれの実験CSVファイルも読み込めませんでした。")
    
    # 複数のDFを縦に結合
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 行動特徴のキーワード
    behaviors = [
        '周囲を見回す回数が多い',
        '進行方向を何度も変えている',
        '歩行速度が早い',
        '歩行中に立ち止まることが多い'
    ]
    
    # 結果を格納するDataFrameと詳細統計用のdict
    results = pd.DataFrame(columns=['行動特徴', 'F値', 'p値', '相関係数', '相関p値', '有意差(ANOVA)', '有意差(相関)'])
    detailed_results = {}
    
    # 「迷子だと感じた」の列を先に全取得（複数ある想定）
    lost_cols = [col for col in combined_df.columns if '迷子になっていると感じた' in col]
    if not lost_cols:
        raise ValueError("データに『迷子になっていると感じた』列が見つかりません。列名を確認してください。")
    
    # 迷子評価データを一つにまとめて1次元に（行列→一次元ベクトル）
    # flattenする前にNaN除去のためにmaskで揃える方法もあるが、
    # 今回はあとでmask処理を行うのでまずはnumpy配列化しておく
    lost_vals_raw = combined_df[lost_cols].values.flatten()

    for behavior in behaviors:
        # 行動特徴にマッチする列を抽出
        behavior_cols = [col for col in combined_df.columns if behavior in col]
        if not behavior_cols:
            print(f"『{behavior}』に該当する列が見つかりません。スキップします。")
            continue
        
        # 該当列をまとめて一次元に
        behavior_vals_raw = combined_df[behavior_cols].values.flatten()
        
        # --- 欠損値のマスクを作成 ---
        mask = (~np.isnan(behavior_vals_raw)) & (~np.isnan(lost_vals_raw))
        behavior_vals = behavior_vals_raw[mask]
        lost_vals = lost_vals_raw[mask]
        
        if len(behavior_vals) == 0:
            print(f"『{behavior}』は有効なデータがありません。")
            continue
        
        # ピアソン相関（相関係数＆相関p値）
        corr_coeff, corr_pval = stats.pearsonr(behavior_vals, lost_vals)
        
        # 行動特徴の中央値で二分割 → High group / Low group
        median_val = np.median(behavior_vals)
        high_group = lost_vals[behavior_vals >= median_val]
        low_group = lost_vals[behavior_vals < median_val]
        
        # 一元分散分析(f_oneway)で高群と低群の差を検定
        f_stat, p_val_anova = stats.f_oneway(high_group, low_group)
        
        # 結果テーブルへ行追加
        results.loc[len(results)] = [
            behavior,
            f_stat,
            p_val_anova,
            corr_coeff,
            corr_pval,
            label_significance(p_val_anova),
            label_significance(corr_pval)
        ]
        
        # 詳細情報
        detailed_results[behavior] = {
            'mean': np.mean(behavior_vals),
            'std': np.std(behavior_vals),
            'median': median_val,
            'correlation': corr_coeff,
            'correlation_pval': corr_pval,
            'f_stat': f_stat,
            'p_value_anova': p_val_anova
        }
    
    return results, detailed_results


def display_results(results):
    """
    「行動特徴」「F値」「p値」「相関係数」「相関p値」などを整形して表示。
    """
    print("\n迷子行動の分析結果:")
    print("=" * 90)
    header = f"{'行動特徴':<28} {'F値':>8} {'p値(ANOVA)':>12} {'相関係数':>10} {'相関p値':>10} {'有意差(ANOVA)':>12} {'有意差(相関)':>12}"
    print(header)
    print("-" * 90)
    
    for _, row in results.iterrows():
        behavior = row['行動特徴']
        f_val = f"{row['F値']:.2f}"
        p_val_anova = f"{row['p値']:.3g}"
        corr = f"{row['相関係数']:.2f}"
        corr_p = f"{row['相関p値']:.3g}"
        sig_anova = row['有意差(ANOVA)']
        sig_corr = row['有意差(相関)']
        
        print(f"{behavior:<30} "
              f"{f_val:>6} "
              f"{p_val_anova:>12} "
              f"{corr:>10} "
              f"{corr_p:>10} "
              f"{sig_anova:>12} "
              f"{sig_corr:>12}")
    
    print("=" * 90)
    print("\n有意差ラベル: * (p<0.05), ** (p<0.01), '' (n.s.)")


def plot_correlation(results):
    """
    行動特徴ごとの「相関係数」を棒グラフで可視化し、p値に応じたアスタリスクを付加する。
    """
    plt.figure(figsize=(10, 6))
    
    # p値に応じて色を変えるサンプル（ANOVAではなく相関p値にしてもOK）
    colors = []
    for p in results['p値']:
        if p < 0.01:
            colors.append('lightcoral')  # p<0.01
        elif p < 0.05:
            colors.append('lightgreen')  # p<0.05
        else:
            colors.append('skyblue')     # p>=0.05
    
    # ただし、上記コードでは results['p値'] が無いとエラーになるので
    # ここは注意：本コードでは ANOVA のp値は 'p値' でなく 'p値(ANOVA)' 列名にするなど要調整
    # あるいは下記のように「相関のp値」で色を変える場合:
    # for p in results['相関p値']:

    # ここでは相関のp値で色を変える例として書き換えます:
    color_list = []
    for p in results['相関p値']:
        if p < 0.01:
            color_list.append('lightcoral')
        elif p < 0.05:
            color_list.append('lightgreen')
        else:
            color_list.append('skyblue')
    
    plt.bar(results['行動特徴'], results['相関係数'], color=color_list)
    plt.xticks(rotation=45, ha='right')
    plt.title('各行動特徴と「迷子だと感じた」評価の相関')
    plt.ylabel('相関係数')
    
    # 棒の上に有意差マークを追加（相関のp値ベースで）
    for i, (r, p) in enumerate(zip(results['相関係数'], results['相関p値'])):
        label = label_significance(p)
        if label:
            plt.text(i, r, label, ha='center', va='bottom', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def display_detailed_results(detailed_results):
    """
    平均・標準偏差・中央値・相関係数などを行動特徴ごとに詳細表示。
    """
    print("\n詳細な統計結果:")
    print("=" * 110)
    header = (
        f"{'行動特徴':<30}"
        f"{'平均':>10}"
        f"{'標準偏差':>10}"
        f"{'中央値':>10}"
        f"{'相関係数':>10}"
        f"{'相関p値':>10}"
        f"{'F値(ANOVA)':>12}"
        f"{'p値(ANOVA)':>12}"
    )
    print(header)
    print("-" * 110)
    
    for behavior, stats_dict in detailed_results.items():
        print(f"{behavior:<30}"
              f"{stats_dict['mean']:>10.2f}"
              f"{stats_dict['std']:>10.2f}"
              f"{stats_dict['median']:>10.2f}"
              f"{stats_dict['correlation']:>10.2f}"
              f"{stats_dict['correlation_pval']:>10.3g}"
              f"{stats_dict['f_stat']:>12.2f}"
              f"{stats_dict['p_value_anova']:>12.3g}")
    
    print("=" * 110)


def main():
    """
    メイン処理:
      1. データの集計・解析 (analyze_multiple_experiments)
      2. 結果の表示 (display_results, display_detailed_results)
      3. 相関のプロット (plot_correlation)
    """
    try:
        results, detailed_results = analyze_multiple_experiments()
        display_results(results)
        display_detailed_results(detailed_results)
        
        # 棒グラフの描画・保存（相関係数）
        plot_correlation(results)
        print("\n分析が完了しました。グラフは 'correlation_plot.png' として保存されました。")

        # (2) ここで 3要因のデータを整備して解析する
        #     例: もし "combined_df" を返すように analyze_multiple_experiments() を
        #         書き換えていたなら、そこからA,B,C,subject,score列を作る。
        #         あるいは別途CSVを読み込み直す。

        # 仮に、"df_3way.csv" というファイルに下記の列があるとする:
        #   subject, A, B, C, score
        #   (1人の被験者が8条件(= A×B×C)を体験したデータ)
        df_3way = pd.read_csv("df_3way.csv")
        
        # 交互作用解析
        anova_result = analyze_three_way_interaction(df_3way)
        # anova_result は AnovaRM.fit() の返り値で、印刷されたテーブルに交互作用(A:B, A:C, B:C, A:B:C)が出力されます。

    except Exception as e:
        print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
