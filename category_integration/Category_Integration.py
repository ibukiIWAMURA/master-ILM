import random

def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()
    form = parts[1].strip()
    return semantic_structure, form


def clustering_rule_set(rule_set):
    word_rule_set = []
    sentence_rule_set = []
    
    for rule in rule_set:
        semantic_structure, _ = parse_rule(rule)
        if semantic_structure.startswith("S/"):
            sentence_rule_set.append(rule)
        else:
            word_rule_set.append(rule)
    
    return word_rule_set, sentence_rule_set

def word_rule_pair(word_rule_set):
    pairs = []
    n = len(word_rule_set)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((word_rule_set[i], word_rule_set[j]))
    return pairs

def detect_word_sim_diff_ability(a_word_rule1, a_word_rule2):
    split_rule1 = split_word_rule(a_word_rule1)
    split_rule2 = split_word_rule(a_word_rule2)
    return (split_rule1[0] != split_rule2[0]) and (split_rule1[1:] == split_rule2[1:])

def can_category_integration_pair_set(pairs):
    integrable_pairs = set()
    for pair in pairs:
        if detect_word_sim_diff_ability(pair[0], pair[1]):
            integrable_pairs.add(pair)
    return integrable_pairs

# def split_word_rule(word_rule):
#     category_label, rest = word_rule.split('/')
#     meaning, form = rest.split('->')
#     return [category_label, meaning, form]


def split_word_rule(word_rule):
    try:
        category_label, rest = word_rule.split('/')
        meaning, form = rest.split('->')
        return [category_label, meaning, form]
    except ValueError as e:
        print(f"Error processing rule: {word_rule}")
        raise e


def category_integration_ability(split_word_rule1, split_word_rule2):
    unified_category_label = random.choice([split_word_rule1[0], split_word_rule2[0]])
    # 選ばれなかった方のカテゴリーラベルを取得
    if unified_category_label == split_word_rule1[0]:
        not_chosen_category_label = split_word_rule2[0]
    else:
        not_chosen_category_label = split_word_rule1[0]
    
    return f"{unified_category_label}/{split_word_rule1[1]}->{split_word_rule1[2]}", unified_category_label, not_chosen_category_label


def replace_category_labels(rule_set, unified_category_label, not_chosen_category_label):
    replaced_rules = []
    for rule in rule_set:
        if f"{not_chosen_category_label}/" in rule:
            # カテゴリーラベルを置換
            replaced_rule = rule.replace(f"{not_chosen_category_label}/", f"{unified_category_label}/", 1)
            replaced_rules.append(replaced_rule)
        else:
            replaced_rules.append(rule)
    return list(set(replaced_rules))

def apply_label_conversion_history(rule_set, label_conversion_history):
    updated_rules = []
    for rule in rule_set:
        for old_label, new_label in label_conversion_history.items():
            rule = rule.replace(f"{old_label}/", f"{new_label}/")
        updated_rules.append(rule)
    return updated_rules

def update_word_and_sentence_rules(word_rule_set, sentence_rule_set, integrable_pairs, label_conversion_history):
    integration_rules = []
    
    for pair in integrable_pairs:
        # 各ペアのルールを分割してカテゴリーラベルを統合
        split_rule1 = split_word_rule(pair[0])
        split_rule2 = split_word_rule(pair[1])
        unified_rule, chosen_label, not_chosen_label = category_integration_ability(split_rule1, split_rule2)
        
        # 統合されたルールをリストに追加
        integration_rules.append(unified_rule)
        
        # ラベルの変換履歴に追加
        label_conversion_history[not_chosen_label] = chosen_label
        
        # 変換履歴を使って単語ルールと文ルールを更新
        word_rule_set = replace_category_labels(word_rule_set, chosen_label, not_chosen_label)
        sentence_rule_set = replace_category_labels(sentence_rule_set, chosen_label, not_chosen_label)
    
    return word_rule_set, sentence_rule_set, integration_rules


def category_integration_learning(rule_set, label_conversion_history):
    
    # ルールセットを変換履歴に基づいて変換
    rule_set = apply_label_conversion_history(rule_set, label_conversion_history)
    
    # 1. 単語ルールと文生成ルールを分類
    word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
    
    # 2. 単語ルールペアを生成
    pairs = word_rule_pair(word_rule_set)
    
    # 3. 統合可能なルールペアを検出
    integrable_pairs = can_category_integration_pair_set(pairs)
    
    category_integration_applications = len(integrable_pairs)  # 適用回数として統合可能ペアの数をカウント

    
    # 4. 統合可能なルールペアから統合ルールを生成・更新
    updated_word_rule_set, updated_sentence_rule_set, integration_rules = update_word_and_sentence_rules(
        word_rule_set, sentence_rule_set, integrable_pairs, label_conversion_history
    )
    
    # 5. 単語ルールと文生成ルールを統合して最終結果を返す
    integrated_rules = updated_word_rule_set + updated_sentence_rule_set
    
    return integrated_rules, category_integration_applications  # 適用回数を一緒に返す