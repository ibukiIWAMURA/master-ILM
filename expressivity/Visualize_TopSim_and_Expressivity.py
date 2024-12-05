import matplotlib.pyplot as plt

def visualize_combined_metrics(topsims, normalized_topsims, ibuki_topsims, expressivities, n_gens, output_path):

    T = list(range(1, n_gens + 1))

    plt.figure(figsize=(10, 6))

    # TopSimを青でプロット
    plt.plot(T, topsims, color='blue', label='TopSim', marker='o')

    # Normalized_TopSimを緑でプロット
    plt.plot(T, normalized_topsims, color='green', label='Normalized_TopSim', marker='x')
    
    # Ibuki_1_TopSimを紫で破線の星形マーカーでプロット
    plt.plot(T, ibuki_topsims, color='purple', linestyle='--', label='Ibuki_1_TopSim', marker='*')

    # Expressivityを赤でプロット
    plt.plot(T, expressivities, color='red', label='Expressivity', marker='s')

    plt.title('Compositionality and Expressivity')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend()

    # x軸のラベルは5刻みで表示
    plt.xticks(T[::5], [str(i) for i in T[::5]])

    # グラフの保存
    plt.savefig(output_path)
    plt.show()

    print(f"グラフを保存しました: {output_path}")

