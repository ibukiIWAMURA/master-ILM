import matplotlib.pyplot as plt

def visualize_operations_number(chunk_counts, category_counts, replace_counts, knowledge_counts, 
                                len_holistic, len_gen_1, len_gen_2, len_gen_3, len_word,
                                n_gens, out_path):
    # Generationsリストに1世代目（0世代目としてカウント）を追加
    generations = list(range(n_gens + 1))  # 1世代目の0を含むように設定

    plt.figure(figsize=(12, 8))
    
    # 各適用回数をプロット
    plt.plot(generations, chunk_counts, label="Chunk", marker="o")
    plt.plot(generations, category_counts, label="Category", marker="o")
    plt.plot(generations, replace_counts, label="Replace", marker="o")
    plt.plot(generations, knowledge_counts, label="Knowledge Count", marker="x")  # 知識数を追加
    
    # 各ルールカテゴリ数のプロット
    plt.plot(generations, len_holistic, label="Holistic Rules", marker="s", linestyle="-")
    plt.plot(generations, len_gen_1, label="Generalization Rules (1 var)", marker="s", linestyle="-")
    plt.plot(generations, len_gen_2, label="Generalization Rules (2 vars)", marker="s", linestyle="-")
    plt.plot(generations, len_gen_3, label="Generalization Rules (3 vars)", marker="s", linestyle="-")
    plt.plot(generations, len_word, label="Word Rules", marker="s", linestyle="-")

    
    # 縦軸の範囲を設定
    plt.ylim(0, max(max(chunk_counts), max(category_counts), max(replace_counts), max(knowledge_counts), 
                    max(len_holistic), max(len_gen_1), max(len_gen_2), max(len_gen_3), max(len_word)) + 5)

    # ラベルと凡例
    plt.xlabel("Generations")
    plt.ylabel("Counts")
    plt.title("3 Operations and Knowledge Counts")
    plt.legend()

    # グラフの保存
    
    plt.savefig(out_path)
    plt.show()

    print(f"グラフを保存しました: {out_path}")
