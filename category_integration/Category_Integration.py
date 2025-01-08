# import random
from randomness.Shared_rng import shared_rng

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

def split_word_rule(word_rule):
    try:
        category_label, rest = word_rule.split('/')
        meaning, form = rest.split('->')
        return [category_label, meaning, form]
    except ValueError as e:
        print(f"Error processing rule: {word_rule}")
        raise e
        
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

def category_integration_ability(split_word_rule1, split_word_rule2):
    unified_category_label = shared_rng.choice([split_word_rule1[0], split_word_rule2[0]])
    # 選ばれなかった方のカテゴリーラベルを取得
    if unified_category_label == split_word_rule1[0]:
        not_chosen_category_label = split_word_rule2[0]
    else:
        not_chosen_category_label = split_word_rule1[0]
    
    return f"{unified_category_label}/{split_word_rule1[1]}->{split_word_rule1[2]}", unified_category_label, not_chosen_category_label

def replace_category_labels(rule_set, unified_category_label, not_chosen_category_label):
    # print('-----')
    # print("選択されたラベル", unified_category_label)
    replaced_rules = []
    # print("取得ルール", rule_set)
    for rule in rule_set:
        # print("選択されないラベル", not_chosen_category_label)
        if f"{not_chosen_category_label}/" in rule:
            # カテゴリーラベルを置換
            replaced_rule = rule.replace(f"{not_chosen_category_label}/", f"{unified_category_label}/", 1)
            # print('変換されたルール', replaced_rule)
            replaced_rules.append(replaced_rule)
        else:
            replaced_rules.append(rule)
    return list(set(replaced_rules))

def update_word_and_sentence_rules(word_rule_set, sentence_rule_set, integrable_pairs):
    integration_rules = []
    # print('関数内：統合されたルール', integration_rules)
    
    # ルールセットを動的に更新するためのリストを用意
    updated_word_rule_set = word_rule_set[:]
    # print("更新されていく単語ルール", updated_word_rule_set)
    updated_sentence_rule_set = sentence_rule_set[:]
    # print("更新されていく文ルール", updated_sentence_rule_set)
    
    for pair in list(integrable_pairs):  # ペアをリストに変換して変更可能に
        # print('統合可能なペアだよ〜', pair)
        if pair[0] not in updated_word_rule_set or pair[1] not in updated_word_rule_set:
            # どちらかがすでに削除されている場合、スキップ
            continue
        
        # 各ペアのルールを分割してカテゴリーラベルを統合
        split_rule1 = split_word_rule(pair[0])
        split_rule2 = split_word_rule(pair[1])
        unified_rule, chosen_label, not_chosen_label = category_integration_ability(split_rule1, split_rule2)
        
        # 統合されたルールをリストに追加
        integration_rules.append(unified_rule)
        # print('統合されていくルールセット', integration_rules)
        
        # 選択されなかったカテゴリーラベルを選択されたものに置換
        updated_word_rule_set = replace_category_labels(updated_word_rule_set, chosen_label, not_chosen_label)
        # print('関数内：更新された単語ルール', updated_word_rule_set)
        updated_sentence_rule_set = replace_category_labels(updated_sentence_rule_set, chosen_label, not_chosen_label)
        # print('関数内：更新された文ルール', updated_sentence_rule_set)
        
        # 統合対象となったルールを削除
        updated_word_rule_set = [rule for rule in updated_word_rule_set if rule != pair[1]]
        
    return updated_word_rule_set, updated_sentence_rule_set, integration_rules

def category_integration_learning(rule_set):
    category_integration_applications = 0
    integrated_rules = rule_set  # 初期ルールセットを integrated_rules として扱う
    while True:
        # 1. 単語ルールと文生成ルールを分類
        word_rule_set, sentence_rules = clustering_rule_set(integrated_rules)
        # print("最終関数：単語・文ルール", word_rule_set, sentence_rules)
        
        # 2. 単語ルールペアを生成
        pairs = word_rule_pair(word_rule_set)
        
        # 3. 統合可能なルールペアを検出
        integrable_pairs = can_category_integration_pair_set(pairs)
        # print("最終関数：統合可能", integrable_pairs)
        
        if not integrable_pairs:
            # 統合可能なペアがない場合、終了
            break

        # 最初のペアのみ取得
        pair = next(iter(integrable_pairs))
        # print("選択されたペア", pair)

        # 統合処理を実行
        category_integration_application = 1
        category_integration_applications += category_integration_application

        updated_word_rule_set, updated_sentence_rule_set, integration_rules = update_word_and_sentence_rules(
            word_rule_set, sentence_rules, {pair}  # 最初の要素だけを処理
        )
        # print('最終関数：更新後の単語・文', updated_word_rule_set, updated_sentence_rule_set, integration_rules)
        
        # 単語ルールと文ルールを統合して integrated_rules を更新
        duplicate_rule_set = updated_word_rule_set + integration_rules + updated_sentence_rule_set
        # print(duplicate_rule_set)
        integrated_rules = list(set(updated_word_rule_set + integration_rules + updated_sentence_rule_set))
        # print('最終関数：ルール全体', integrated_rules)
    
    # print("カテゴリー統合の結果", integrated_rules)
    return integrated_rules, category_integration_applications