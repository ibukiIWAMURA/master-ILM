import matplotlib.pyplot as plt

def visualize_operations_number(chunk_counts, category_counts, replace_counts, knowledge_counts, 
                                len_holistic, len_gen_1, len_gen_2, len_gen_3, len_word,
                                n_gens, out_paths, n_samples):
    # 初期値の設定
    initial_value = n_samples * 2
    if len_holistic:
        len_holistic[0] = initial_value
    if knowledge_counts:
        knowledge_counts[0] = initial_value
        
    generations = list(range(n_gens + 1))  # 世代のリスト
    
    # グラフ1: 学習アルゴリズムと言語知識の総数
    plt.figure(figsize=(12, 8))
    plt.plot(generations, chunk_counts, label="Chunk", color="red")
    plt.plot(generations, category_counts, label="Category", color="blue")
    plt.plot(generations, replace_counts, label="Replace", color="green")
    plt.plot(generations, knowledge_counts, label="Knowledge Count", color="black")
    plt.xlabel("Generations")
    plt.ylabel("Counts")
    plt.ylim((0, 150))
    plt.title("3 Learnings and Knowledge Counts")
    plt.legend()
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(out_paths[0])
    plt.close()

    # グラフ2: 言語知識の中身を積み上げ面グラフ
    plt.figure(figsize=(12, 8))
    cumulative_data = [len_holistic, len_word, len_gen_1, len_gen_2, len_gen_3]
    labels = ["Holistic Rules", "Word Rules", "Generalization Rules (1 var)", 
              "Generalization Rules (2 vars)", "Generalization Rules (3 vars)"]
    colors = ["red", "blue", "green", "purple", "orange"]
    plt.stackplot(generations, *cumulative_data, labels=labels, colors=colors, alpha=0.8)
    plt.xlabel("Generations")
    plt.ylabel("Counts")
    plt.title("Composition of Knowledge")
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(out_paths[1])
    plt.close()

    # グラフ3: 積み上げ面グラフ＋言語知識の総数（棒グラフ）
    plt.figure(figsize=(12, 8))
    plt.plot(generations, knowledge_counts, label="Knowledge Count", color="black", linestyle='-', linewidth=2)
    plt.stackplot(generations, *cumulative_data, labels=labels, colors=colors, alpha=0.8)
    plt.xlabel("Generations")
    plt.ylabel("Counts")
    plt.title("Knowledge Counts and Composition")
    plt.legend(loc="upper left")
    plt.grid(axis='y', linestyle="--", alpha=0.7)
    plt.savefig(out_paths[2])
    plt.close()