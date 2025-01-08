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
    output_path_pearson_metrics,            # 1. Pearson 相関係数のパス
    output_path_spearman_metrics,           # 2. Spearman 相関係数のパス
    output_path_pearson_ibuki_metrics,      # 3. Pearson Ibuki のパス
    output_path_spearman_ibuki_metrics,     # 4. Spearman Ibuki のパス
    output_path_ibuki_metrics,              # 5. Ibuki の分散と平均
    output_path_inference,                  # 6. Inference Accuracy のパス
    output_path_ibuki_index_variance_similar_pearson,  # Index 分散 (Pearson)
    output_path_ibuki_index_average_similar_pearson,  # Index 平均 (Pearson)
    output_path_ibuki_set_variance_difference_pearson, # Set 分散 (Pearson)
    output_path_ibuki_set_average_difference_pearson,  # Set 平均 (Pearson)
    output_path_ibuki_index_variance_similar_spearman,  # Index 分散 (Spearman)
    output_path_ibuki_index_average_similar_spearman,  # Index 平均 (Spearman)
    output_path_ibuki_set_variance_difference_spearman, # Set 分散 (Spearman)
    output_path_ibuki_set_average_difference_spearman  # Set 平均 (Spearman)
):
    
    T = list(range(1, n_gens + 1))
    step = max(1, n_gens // 10)  # 横軸を最大10分割

    # 1. ピアソン相関係数＋表現度
    plt.figure(figsize=(12, 8))

    # TopSim (cv=0とcv=1) のプロット
    plt.plot(T, topsims_cv0, color='red', label='TopSim CV=0')
    plt.plot(T, topsims_cv1, color='red', linestyle='--', label='TopSim CV=1')

    # Normalized_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, normalized_topsims_cv0, color='blue', label='Normalized_TopSim CV=0')
    plt.plot(T, normalized_topsims_cv1, color='blue', linestyle='--', label='Normalized_TopSim CV=1')

    # Ibuki_1_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, ibuki_topsims_cv0, color='green', label='Ibuki_1_TopSim CV=0')
    plt.plot(T, ibuki_topsims_cv1, color='green', linestyle='--', label='Ibuki_1_TopSim CV=1')

    # 最終世代の値を補完
    expressivities_cv0 = fill_missing_values(expressivities_cv0)
    expressivities_cv1 = fill_missing_values(expressivities_cv1)
    
    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='black', label='Expressivity CV=0')
    plt.plot(T, expressivities_cv1, color='black', linestyle='--', label='Expressivity CV=1')

    # グラフタイトルとラベル設定
    plt.title('Compositionality and Expressivity by CV / pearson')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_pearson_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_pearson_metrics}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 2. スピアマン相関係数＋表現度
    plt.figure(figsize=(12, 8))

    # Spearman_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_topsims_cv0, color='red', label='Spearman_TopSim CV=0')
    plt.plot(T, spearman_topsims_cv1, color='red', linestyle='--', label='Spearman_TopSim CV=1')

    # Spearman_Normalized_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_normalized_topsims_cv0, color='blue', label='Spearman_Normalized_TopSim CV=0')
    plt.plot(T, spearman_normalized_topsims_cv1, color='blue', linestyle='--', label='Spearman_Normalized_TopSim CV=1')

    # Spearman_Ibuki_1_TopSim (cv=0とcv=1) のプロット
    plt.plot(T, spearman_ibuki_topsims_cv0, color='green', label='Spearman_Ibuki_1_TopSim CV=0')
    plt.plot(T, spearman_ibuki_topsims_cv1, color='green', linestyle='--', label='Spearman_Ibuki_1_TopSim CV=1')

    # 最終世代の値を補完
    expressivities_cv0 = fill_missing_values(expressivities_cv0)
    expressivities_cv1 = fill_missing_values(expressivities_cv1)
    
    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='black', label='Expressivity CV=0')
    plt.plot(T, expressivities_cv1, color='black', linestyle='--', label='Expressivity CV=1')

    # グラフタイトルとラベル設定
    plt.title('Compositionality and Expressivity by CV / spearman')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_spearman_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_spearman_metrics}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 3. ピアソン相関係数でのイブキ計算
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, ibuki_2_values, color='red', label='Ibuki_2_TopSim')
    
    # Ibuki_3_TopSim のプロット
    plt.plot(T, ibuki_3_values, color='blue', label='Ibuki_3_TopSim')

    # 最終世代の値を補完
    expressivities_cv0 = fill_missing_values(expressivities_cv0)
    expressivities_cv1 = fill_missing_values(expressivities_cv1)
    
    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='green', label='Expressivity CV=0')
    plt.plot(T, expressivities_cv1, color='green', linestyle='--', label='Expressivity CV=1')

    # グラフタイトルとラベル設定
    plt.title('Pearson & Ibuki:Compositionality and Expressivity')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_pearson_ibuki_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_pearson_ibuki_metrics}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 4. スピアマン相関係数でのイブキ計算
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, spearman_ibuki_2_topsim_values, color='red', label='Spearman_Ibuki_2_TopSim')
    
    # Ibuki_3_TopSim のプロット
    plt.plot(T, spearman_ibuki_3_topsim_values, color='blue', label='Spearman_Ibuki_3_TopSim')

    # 最終世代の値を補完
    expressivities_cv0 = fill_missing_values(expressivities_cv0)
    expressivities_cv1 = fill_missing_values(expressivities_cv1)
    
    # Expressivity (cv=0とcv=1) のプロット
    plt.plot(T, expressivities_cv0, color='green', label='Expressivity CV=0')
    plt.plot(T, expressivities_cv1, color='green', linestyle='--', label='Expressivity CV=1')

    # グラフタイトルとラベル設定
    plt.title('Spearman & Ibuki:Compositionality and Expressivity')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_spearman_ibuki_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_spearman_ibuki_metrics}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 5. イブキの平均と分散
    plt.figure(figsize=(12, 8))

    # Ibuki_4_TopSim Variance and Average のプロット
    plt.plot(T, ibuki_4_variances, color='red', label='Ibuki_4_TopSim Variance')
    plt.plot(T, ibuki_4_averages, color='red', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Variance and Average のプロット
    plt.plot(T, ibuki_5_variances, color='blue', label='Ibuki_5_TopSim Variance')
    plt.plot(T, ibuki_5_averages, color='blue', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Ibuki:Variance and Average')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_metrics)
    # plt.show()

    # print(f"Compositionality and Expressivityグラフを保存しました: {output_path_ibuki_metrics}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    
    
    # 6. Inference Accuracy グラフ
    plt.figure(figsize=(12, 8))

    plt.plot(T, inference_accuracy_values, color='red', label='Inference Accuracy')

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
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    
    
    # 7_0. index単位で類似な構造を計算：　分散：ピアソン
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, ibuki_2_values, color='red', label='Ibuki_2_TopSim')

    # Ibuki_4_TopSim Variance
    plt.plot(T, ibuki_4_variances, color='blue', label='Ibuki_4_TopSim Variance')

    # Ibuki_5_TopSim Variance
    plt.plot(T, ibuki_5_variances, color='green', label='Ibuki_5_TopSim Variance')

    # グラフタイトルとラベル設定
    plt.title('Index, variance:Similar structure, pearson')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_index_variance_similar_pearson)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_index_variance_similar_pearson}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 7_1. index単位で類似な構造を計算：　分散：スピアマン
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, spearman_ibuki_2_topsim_values, color='red', label='Spearman_Ibuki_2_TopSim')

    # Ibuki_4_TopSim Variance
    plt.plot(T, ibuki_4_variances, color='blue', label='Ibuki_4_TopSim Variance')

    # Ibuki_5_TopSim Variance
    plt.plot(T, ibuki_5_variances, color='green', label='Ibuki_5_TopSim Variance')

    # グラフタイトルとラベル設定
    plt.title('Index, variance:Similar structure, spearman')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_index_variance_similar_spearman)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_index_variance_similar_spearman}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    # 7.5_0.  index単位で類似な構造を計算：　平均 ：ピアソン
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, ibuki_2_values, color='red', label='Ibuki_2_TopSim')

    # Ibuki_4_TopSim Average
    plt.plot(T, ibuki_4_averages, color='blue', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Average
    plt.plot(T, ibuki_5_averages, color='green', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Index, average:Similar structure, pearson')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_index_average_similar_pearson)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_index_average_similar_pearson}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------

    
    
    # 7.5_1.  index単位で類似な構造を計算：　平均:：スピアマン
    plt.figure(figsize=(12, 8))

    # Ibuki_2_TopSim のプロット
    plt.plot(T, spearman_ibuki_2_topsim_values, color='red', label='Spearman_Ibuki_2_TopSim')

    # Ibuki_4_TopSim Average
    plt.plot(T, ibuki_4_averages, color='blue', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Average
    plt.plot(T, ibuki_5_averages, color='green', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Index, average:Similar structure, spearman')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_index_average_similar_spearman)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_index_average_similar_spearman}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------
    
    # 8_0. set単位で類似な構造を計算: 分散：ピアソン
    plt.figure(figsize=(12, 8))

    # Ibuki_3_TopSim のプロット
    plt.plot(T, ibuki_3_values, color='red', label='Ibuki_3_TopSim')

    # Ibuki_4_TopSim Variance
    plt.plot(T, ibuki_4_variances, color='blue', label='Ibuki_4_TopSim Variance')

    # Ibuki_5_TopSim Variance
    plt.plot(T, ibuki_5_variances, color='green', label='Ibuki_5_TopSim Variance')

    # グラフタイトルとラベル設定
    plt.title('Set, variance:Similar structure, pearson')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_set_variance_difference_pearson)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_set_variance_difference_pearson}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------
    
    # 8_1. set単位で類似な構造を計算: 分散：スピアマン
    plt.figure(figsize=(12, 8))

    # Ibuki_3_TopSim のプロット
    plt.plot(T, spearman_ibuki_3_topsim_values, color='red', label='Spearman_Ibuki_3_TopSim')

    # Ibuki_4_TopSim Variance
    plt.plot(T, ibuki_4_variances, color='blue', label='Ibuki_4_TopSim Variance')

    # Ibuki_5_TopSim Variance
    plt.plot(T, ibuki_5_variances, color='green', label='Ibuki_5_TopSim Variance')

    # グラフタイトルとラベル設定
    plt.title('Set, variance:Similar structure, spearman')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_set_variance_difference_spearman)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_set_variance_difference_spearman}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------
    
    
    # 8.5_0. set単位で類似な構造を計算: 平均：ピアソン
    plt.figure(figsize=(12, 8))

    # Ibuki_3_TopSim のプロット
    plt.plot(T, ibuki_3_values, color='red', label='Ibuki_3_TopSim')

    # Ibuki_4_TopSim Average
    plt.plot(T, ibuki_4_averages, color='blue', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Average
    plt.plot(T, ibuki_5_averages, color='green', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Set, average:Similar structure, pearson')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_set_average_difference_pearson)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_set_average_difference_pearson}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------
    
    # 8.5_1. set単位で類似な構造を計算: 平均：スピアマン
    plt.figure(figsize=(12, 8))

    # Ibuki_3_TopSim のプロット
    plt.plot(T, spearman_ibuki_3_topsim_values, color='red', label='Spearman_Ibuki_3_TopSim')

    # Ibuki_4_TopSim Average
    plt.plot(T, ibuki_4_averages, color='blue', linestyle='--', label='Ibuki_4_TopSim Average')

    # Ibuki_5_TopSim Average
    plt.plot(T, ibuki_5_averages, color='green', linestyle='--', label='Ibuki_5_TopSim Average')

    # グラフタイトルとラベル設定
    plt.title('Set, average:Similar structure, spearman')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.ylim((-0.25, 1))  # y軸の範囲を統一
    plt.legend(loc='upper right')

    # x軸のラベルを調整
    plt.xticks(T[::step], [str(i) for i in T[::step]])

    # グラフの保存
    plt.savefig(output_path_ibuki_set_average_difference_spearman)
    # plt.show()

    # print(f"Ibuki Metricsグラフを保存しました: {output_path_ibuki_set_average_difference_spearman}")
    plt.close()
    # ------------------------------------------------------------------------------------------------------------------------
    