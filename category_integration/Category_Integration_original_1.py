import random
import re

def set_only_word_rule(rule_set):
    # 単語ルールを抽出する正規表現パターン
    pattern = re.compile(r'^[A-Z]/\w+ -> \w+$')
    
    # 単語ルールを抽出
    only_word_rule_set = [rule for rule in rule_set if pattern.match(rule)]
    
    return only_word_rule_set

def word_rule_pair(only_word_rule_set):
    pairs = []
    n = len(only_word_rule_set)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((only_word_rule_set[i], only_word_rule_set[j]))
    return pairs

def split_word_rule(word_rule):
    # "/" と " -> "で文字列を分割
    category_label, rest = word_rule.split('/')
    meaning, form = rest.split(' -> ')
    return [category_label, meaning, form]

def detect_word_sim_diff_ability(a_word_rule1, a_word_rule2):
    split_rule1 = split_word_rule(a_word_rule1)
    split_rule2 = split_word_rule(a_word_rule2)
    
    # カテゴリーラベルが異なり、意味表現と形式表現が共通か確認
    return (split_rule1[0] != split_rule2[0]) and (split_rule1[1:] == split_rule2[1:])

def can_category_integration_pair_set(pairs):
    integrable_pairs = set()
    for pair in pairs:
        if detect_word_sim_diff_ability(pair[0], pair[1]):
            integrable_pairs.add(pair)
    return integrable_pairs

def category_integration_ability(split_word_rule1, split_word_rule2):
    # カテゴリーラベルをランダムに選択
    unified_category_label = random.choice([split_word_rule1[0], split_word_rule2[0]])
    return f"{unified_category_label}/{split_word_rule1[1]} -> {split_word_rule1[2]}"

def category_integration_process(integrable_pairs):
    integrated_rules = []
    for pair in integrable_pairs:
        split_rule1 = split_word_rule(pair[0])
        split_rule2 = split_word_rule(pair[1])
        # 統一したルールを生成
        unified_rule = category_integration_ability(split_word_rule1=split_rule1, split_word_rule2=split_rule2)
        integrated_rules.append(unified_rule)
    return integrated_rules

# すべてのルールのペアを生成
only_word_rule_set = set_only_word_rule(rule_set)
pairs = word_rule_pair(only_word_rule_set)

# 学習アルゴリズムに適用可能なペアを取得
integrable_pairs = can_category_integration_pair_set(pairs)

# ルールセット全体をコピーして保持
remaining_rules = rule_set[:]

# integrable_pairs に含まれるルールを used_rules に追加
used_rules = []
for rule_pair in integrable_pairs:
    for rule in rule_pair:
        # 元の rule_set にあるかを確認し、あれば used_rules に追加
        for original_rule in rule_set:
            if rule in original_rule:
                used_rules.append(original_rule)

# 未使用のルールを特定
unapplied_rules = [rule for rule in remaining_rules if rule not in used_rules]

# 学習アルゴリズムによる統合を実行
integrated_rules = category_integration_process(integrable_pairs)

# 統合されたルールと適用できなかったルールを結合
integrated_rules.extend(unapplied_rules)