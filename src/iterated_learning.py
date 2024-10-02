import random
import os
import chunk.Chunk as Chunk
import category_integration.Category_Integration as CategoryIntegration
import replace.Replace as Replace
import production.Production as Production

class Agent:
    def __init__(self, name, learning_algorithms=None):
        self.name = name
        self.memory = []
        self.learning_algorithms = learning_algorithms if learning_algorithms else ['chunk', 'category_integration', 'replace']

    def produce_language(self, meaning):
        # 発話アルゴリズムを用いて形式表現を生成する
        if meaning in self.memory:
            form = self.memory[meaning]
        else:
            form = Production.generate_form(meaning, self.memory)
        language = f"{meaning} -> {form}"
        print(f"{self.name}発話: {language}")
        return meaning, form

    def learn_language(self, rule_set):
        for algorithm in self.learning_algorithms:
            if algorithm == 'chunk':
                rule_set = Chunk.chunk_learning(rule_set)
            elif algorithm == 'category_integration':
                rule_set = CategoryIntegration.category_integration_learning(rule_set)
            elif algorithm == 'replace':
                rule_set = Replace.replace_learning(rule_set)
        self.memory = rule_set  # 最終的な学習結果を記憶

def sample_meaning_space(file_path, n_samples):
    with open(file_path, 'r') as f:
        meanings = f.readlines()
    return random.sample(meanings, n_samples)

def simulate_language_evolution(generations=10, n_samples=50, initial_language_file='data/100_Initial_Language.txt', semantic_space_file='data/100_Semantic_Space.txt'):
    # エージェントの初期化
    parent = Agent(name="Parent", learning_algorithms=['chunk', 'category_integration', 'replace'])
    child = Agent(name="Child", learning_algorithms=['chunk', 'category_integration', 'replace'])

    for generation in range(1, generations + 1):
        if generation == 1:
            # 最初の世代は初期言語を使って発話
            sampled_meanings = sample_meaning_space(initial_language_file, n_samples)
            parent.memory = [f"{meaning.strip()} -> {Production.generate_initial_form(meaning.strip())}" for meaning in sampled_meanings]
        else:
            # 各世代で発話指令としての意味表現をランダムにサンプリング
            sampled_meanings = sample_meaning_space(semantic_space_file, n_samples)

            # 親エージェントがサンプリングした意味を発話
            for meaning in sampled_meanings:
                language = parent.produce_language(meaning.strip())
                # 子エージェントがその言語を学習
                child.learn_language(parent.memory)

        # 学習結果を保存して発話フェーズに進む
        output_file = f"out/generation_{generation}_output.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for rule in child.memory:
                f.write(rule + "\n")

        # 世代交代
        parent, child = child, Agent(name="Child", learning_algorithms=['chunk', 'category_integration', 'replace'])

if __name__ == "__main__":
    simulate_language_evolution(10)
