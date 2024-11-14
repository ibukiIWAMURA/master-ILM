# import matplotlib.pyplot as plt

# def visualize_combined_metrics(topsims, normalized_topsims, ibuki_topsims, expressivities, n_gens, output_path):

#     T = list(range(1, n_gens + 1))

#     plt.figure(figsize=(10, 6))

#     # TopSimを青でプロット
#     plt.plot(T, topsims, color='blue', label='TopSim', marker='o')

#     # Normalized_TopSimを緑でプロット
#     plt.plot(T, normalized_topsims, color='green', label='Normalized_TopSim', marker='x')
    
#     # Ibuki_1_TopSimを紫で破線の星形マーカーでプロット
#     plt.plot(T, ibuki_topsims, color='purple', linestyle='--', label='Ibuki_1_TopSim', marker='*')

#     # Expressivityを赤でプロット
#     plt.plot(T, expressivities, color='red', label='Expressivity', marker='s')

#     plt.title('Compositionality and Expressivity')
#     plt.xlabel('Generation')
#     plt.ylabel('Value')
#     plt.ylim((-0.25, 1))  # y軸の範囲を統一
#     plt.legend()

#     # x軸のラベルは5刻みで表示
#     plt.xticks(T[::5], [str(i) for i in T[::5]])

#     # グラフの保存
#     plt.savefig(output_path)
#     plt.show()

#     print(f"グラフを保存しました: {output_path}")


import matplotlib.pyplot as plt

def visualize_combined_metrics(
    topsims_cv0, topsims_cv1, 
    normalized_topsims_cv0, normalized_topsims_cv1, 
    ibuki_topsims_cv0, ibuki_topsims_cv1, 
    expressivities_cv0, expressivities_cv1, 
    n_gens, output_path
):
    T = list(range(1, n_gens + 1))

    plt.figure(figsize=(12, 8))

    # TopSim (cv=0とcv=1) のプロット
    plt.plot(T, topsims_cv0, color='blue', label='TopSim CV=0', marker='o')
    plt.plot(T, topsims_cv1, color='blue', linestyle='--', label='TopSim CV=1', marker='x')

    # Normalized_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, normalized_topsims_cv0, color='green', label='Normalized_TopSim CV=0', marker='s')
    plt.plot(T, normalized_topsims_cv1, color='green', linestyle='--', label='Normalized_TopSim CV=1', marker='^')

    # Ibuki_1_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, ibuki_topsims_cv0, color='purple', label='Ibuki_1_TopSim CV=0', marker='D')
    plt.plot(T, ibuki_topsims_cv1, color='purple', linestyle='--', label='Ibuki_1_TopSim CV=1', marker='*')

    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='red', label='Expressivity CV=0', marker='p')
    plt.plot(T, expressivities_cv1, color='red', linestyle='--', label='Expressivity CV=1', marker='v')

    # グラフタイトルとラベル設定
    plt.title('Compositionality and Expressivity by CV Values')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルは5刻みで表示
    plt.xticks(T[::5], [str(i) for i in T[::5]])

    # グラフの保存
    plt.savefig(output_path)
    plt.show()

    print(f"グラフを保存しました: {output_path}")