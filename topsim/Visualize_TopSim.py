import matplotlib.pyplot as plt
import os

def visualize_topsim(topsims, n_gens, out_path):
    # 世代数はtopsimsの長さ
    n_gens = len(topsims)
    T = list(range(1, n_gens + 1))  # プロット位置は1から開始
    
    # グラフの描画
    plt.figure(figsize=(10, 6))
    plt.ylim((-0.25, 1))
    plt.plot(T, topsims)
    plt.title('TopSims')
    plt.xlabel('Generation')
    plt.ylabel('TopSim')
    
    # x軸のラベルは5刻みで表示
    plt.xticks(T[::5], [str(i) for i in T[::5]])  
    
    # グラフを保存
    plt.savefig(out_path)
    
    # グラフを表示
    plt.show()

    print(f"グラフを保存しました: {out_path}")
