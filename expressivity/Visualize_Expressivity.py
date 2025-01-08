import matplotlib.pyplot as plt

def visualize_expressivity(expressivity_values, n_gens, out_path):
    # 1世代目のExpressivity値を0に設定
    expressivity_values = [0] + expressivity_values  # 1世代目を0に

    # プロット位置は1から n_gens まで
    T = list(range(1, n_gens + 1))

    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.ylim((-0.25, 1))
    plt.plot(T, expressivity_values)
    plt.title('Expressivity')
    plt.xlabel('Generation')
    plt.ylabel('Expressivity')

    # x軸のラベルは5刻みで表示
    plt.xticks([1] + T[::5], [str(i) for i in [1] + T[::5]])  
    
    # グラフを保存
    plt.savefig(out_path)
    
    # グラフを表示
    plt.show()

    print(f"グラフを保存しました: {out_path}")