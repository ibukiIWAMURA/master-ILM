import matplotlib.pyplot as plt

def fill_missing_values(data_list):
    """最終世代の値が空の場合、直前の値を繰り返す"""
    if not data_list[-1]:  # 最終値が空の場合
        data_list[-1] = data_list[-2]  # 直前の値を使用
    return data_list

def visualize_combined_metrics(
    topsims_cv0, topsims_cv1,
    spearman_topsims_cv0, spearman_topsims_cv1,
    normalized_topsims_cv0, normalized_topsims_cv1, 
    spearman_normalized_topsims_cv0, spearman_normalized_topsims_cv1,
    ibuki_topsims_cv0, ibuki_topsims_cv1, 
    spearman_ibuki_topsims_cv0, spearman_ibuki_topsims_cv1, 
    expressivities_cv0, expressivities_cv1, 
    inference_accuracy_values,
    ibuki_2_values, 
    spearman_ibuki_2_topsim_values,
    ibuki_3_values,
    spearman_ibuki_3_topsim_values,
    ibuki_4_variances, ibuki_4_averages,
    ibuki_5_variances, ibuki_5_averages,
    n_gens, 
    output_path_metrics, output_path_inference, output_path_ibuki
):
    
    T = list(range(1, n_gens + 1))
    step = max(1, n_gens // 10)  # 横軸を最大10分割

    # 1. Compositionality and Expressivity グラフ
    plt.figure(figsize=(12, 8))

    # TopSim (cv=0とcv=1) のプロット
    plt.plot(T, topsims_cv0, color='blue', label='TopSim CV=0')
    plt.plot(T, topsims_cv1, color='blue', linestyle='--', label='TopSim CV=1')
    # Spearman_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_topsims_cv0, color='orange', label='Spearman_TopSim CV=0')
    plt.plot(T, spearman_topsims_cv1, color='orange', linestyle='--', label='Spearman_TopSim CV=1')

    # Normalized_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, normalized_topsims_cv0, color='green', label='Normalized_TopSim CV=0')
    plt.plot(T, normalized_topsims_cv1, color='green', linestyle='--', label='Normalized_TopSim CV=1')
    # Spearman_Normalized_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_normalized_topsims_cv0, color='yellow', label='Spearman_Normalized_TopSim CV=0')
    plt.plot(T, spearman_normalized_topsims_cv1, color='yellow', linestyle='--', label='Spearman_Normalized_TopSim CV=1')

    # Ibuki_1_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, ibuki_topsims_cv0, color='purple', label='Ibuki_1_TopSim CV=0')
    plt.plot(T, ibuki_topsims_cv1, color='purple', linestyle='--', label='Ibuki_1_TopSim CV=1')
    # Spearman_Ibuki_1_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_ibuki_topsims_cv0, color='black', label='Spearman_Ibuki_1_TopSim CV=0')
    plt.plot(T, spearman_ibuki_topsims_cv1, color='black', linestyle='--', label='Spearman_Ibuki_1_TopSim CV=1')

    # 最終世代の値を補完
    expressivities_cv0 = fill_missing_values(expressivities_cv0)
    expressivities_cv1 = fill_missing_values(expressivities_cv1)
    
    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='red', label='Expressivity CV=0')
    plt.plot(T, expressivities_cv1, color='red', linestyle='--', label='Expressivity CV=1')

    # グラフタイトルとラベル設定
    plt.title('Compositionality and Expressivity by CV Values')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_metrics}")
    # ------------------------------------------------------------------------------------------------------------------------

    # 2. Inference Accuracy グラフ
    plt.figure(figsize=(12, 6))

    plt.plot(T, inference_accuracy_values, color='orange', label='Inference Accuracy')

    # グラフタイトルとラベル設定
    plt.title('Inference Accuracy by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='lower right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_inference)
    # plt.show()

    # print(f"Inference Accuracyグラフを保存しました: {output_path_inference}")
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 3. Ibuki Metrics グラフ
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, ibuki_2_values, color='blue', label='Ibuki_2_TopSim')
    plt.plot(T, spearman_ibuki_2_topsim_values, color='black', label='Spearman_Ibuki_2_TopSim')
    

    # Ibuki_3_TopSim のプロット
    plt.plot(T, ibuki_3_values, color='green', label='Ibuki_3_TopSim')
    plt.plot(T, spearman_ibuki_3_topsim_values, color='gray', label='Spearman_Ibuki_3_TopSim')

    # Ibuki_4_TopSim Variance and Average のプロット
    plt.plot(T, ibuki_4_variances, color='red', label='Ibuki_4_TopSim Variance')
    plt.plot(T, ibuki_4_averages, color='red', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Variance and Average のプロット
    plt.plot(T, ibuki_5_variances, color='cyan', label='Ibuki_5_TopSim Variance')
    plt.plot(T, ibuki_5_averages, color='cyan', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Ibuki Metrics by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki}")
