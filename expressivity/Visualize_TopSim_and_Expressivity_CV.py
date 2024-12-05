import matplotlib.pyplot as plt

def visualize_combined_metrics(
    topsims_cv0, topsims_cv1, 
    normalized_topsims_cv0, normalized_topsims_cv1, 
    ibuki_topsims_cv0, ibuki_topsims_cv1, 
    expressivities_cv0, expressivities_cv1, 
    inference_accuracy_values,
    n_gens, output_path_metrics, output_path_inference
):
    T = list(range(1, n_gens + 1))

    # 1. Compositionality and Expressivity グラフ
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
    plt.savefig(output_path_metrics)
    plt.show()

    print(f"Compositionality and Expressivityグラフを保存しました: {output_path_metrics}")

    # 2. Inference Accuracy グラフ
    plt.figure(figsize=(12, 6))

    plt.plot(T, inference_accuracy_values, color='orange', label='Inference Accuracy', marker='o')

    # グラフタイトルとラベル設定
    plt.title('Inference Accuracy by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.ylim((0, 1))  # y軸の範囲を0-1に固定
    plt.legend(loc='lower right')

    # x軸のラベルは5刻みで表示
    plt.xticks(T[::5], [str(i) for i in T[::5]])

    # グラフの保存
    plt.savefig(output_path_inference)
    plt.show()

    print(f"Inference Accuracyグラフを保存しました: {output_path_inference}")
