import matplotlib.pyplot as plt

def visualize_operations_number(chunk_counts, category_counts, replace_counts, knowledge_counts, n_gens, out_path):
    # Generationsリストに1世代目（0世代目としてカウント）を追加
    generations = list(range(n_gens + 1))  # 1世代目の0を含むように設定

    plt.figure(figsize=(10, 6))
    
    # 各適用回数をプロット
    plt.plot(generations, chunk_counts, label="Chunk", marker="o")
    plt.plot(generations, category_counts, label="Category", marker="o")
    plt.plot(generations, replace_counts, label="Replace", marker="o")
    plt.plot(generations, knowledge_counts, label="Knowledge Count", marker="x")  # 知識数を追加
    
    # 縦軸の範囲を設定
    plt.ylim(0, max(max(chunk_counts), max(category_counts), max(replace_counts), max(knowledge_counts)) + 5)

    # ラベルと凡例
    plt.xlabel("Generations")
    plt.ylabel("Counts")
    plt.title("3 Operations and Knowledge Counts")
    plt.legend()

    # グラフの保存
    plt.savefig(out_path)
    plt.show()

    print(f"グラフを保存しました: {out_path}")
