import re

def split_word_rule_set_and_sentence_rule_set(rule_set):
    only_word_rule_set = []
    only_sentence_rule_set = []
    
    for a_rule in rule_set:
        if "S/_" in a_rule:
            only_sentence_rule_set.append(a_rule)
        else:
            only_word_rule_set.append(a_rule)
        
    return only_word_rule_set, only_sentence_rule_set

def pairing_index_a_word_rule_and_a_sentence_rule(only_word_rule_set, only_sentence_rule_set):
    set_index_pair_a_word_rule_and_a_sentence_rule = []
    
    for word_rule_index, a_word_rule in enumerate(only_word_rule_set):
        for sentence_rule_index, a_sentence_rule in enumerate(only_sentence_rule_set):
            set_index_pair_a_word_rule_and_a_sentence_rule.append((word_rule_index, sentence_rule_index))
    
    return set_index_pair_a_word_rule_and_a_sentence_rule

def split_word_rule_ability(a_word_rule):
    # "/" と " -> "で文字列を分割
    category_label, rest = a_word_rule.split('/')
    sem, form_express = rest.split(' -> ')
    sem_express = sem.strip('_')
    split_word_rule_list = [category_label, sem_express, form_express]
    return split_word_rule_list

def split_sentence_rule_ability(a_sentence_rule):
    parts = a_sentence_rule.split('->')
    sem_express = parts[0].strip()
    form_express = parts[1].strip()
    split_sem_express = [re.sub(r'^_', '', part) for part in re.findall(r'_[a-zA-Z0-9]+|\(\w+\)|[A-Z]+|[0-9]', sem_express)]
    split_sentence_rule_list = [(split_sem_express[0], *split_sem_express[1:]), form_express]
    return split_sentence_rule_list

def set_pair_a_word_rule_and_a_sentence_rule_process(only_word_rule_set, only_sentence_rule_set, set_index_pair_a_word_rule_and_a_sentence_rule):
    set_pair_a_word_rule_and_a_sentence_rule = []
    for word_index, sentence_index in set_index_pair_a_word_rule_and_a_sentence_rule:
        a_word_rule = only_word_rule_set[word_index]
        a_sentence_rule = only_sentence_rule_set[sentence_index]
        set_pair_a_word_rule_and_a_sentence_rule.append((a_word_rule, a_sentence_rule))
    return set_pair_a_word_rule_and_a_sentence_rule

def split_set_pair_a_word_rule_and_a_sentence_rule_process(set_pair_a_word_rule_and_a_sentence_rule):
    split_set_pair_a_word_rule_and_a_sentence_rule = []
    for a_word_rule, a_sentence_rule in set_pair_a_word_rule_and_a_sentence_rule:
        split_word_rule = split_word_rule_ability(a_word_rule)
        split_sentence_rule = split_sentence_rule_ability(a_sentence_rule)
        split_set_pair_a_word_rule_and_a_sentence_rule.append((split_word_rule, split_sentence_rule))
    return split_set_pair_a_word_rule_and_a_sentence_rule

def detect_word_sem_in_sentence_sem_and_index_ability(word_rule_for_replace1, sentence_rule_for_replace1):
    sem_in_word_rule_for_replace1 = word_rule_for_replace1[1]
    sem_in_sentence_rule_for_replace1 = sentence_rule_for_replace1[0]
    
    if sem_in_word_rule_for_replace1 in sem_in_sentence_rule_for_replace1:
        index_at_sem_in_sentence_rule_for_replace1 = sem_in_sentence_rule_for_replace1.index(sem_in_word_rule_for_replace1)
        return word_rule_for_replace1, sentence_rule_for_replace1, [1, 0, index_at_sem_in_sentence_rule_for_replace1]
    else:
        return []

def detect_and_split_word_form_in_sentence_form_and_index_ability(word_sem_in_sentence_sem_and_index):
    form_in_word_rule_for_replace1 = word_sem_in_sentence_sem_and_index[0][2]
    form_in_sentence_rule_for_replace1 = word_sem_in_sentence_sem_and_index[1][1]
    
    if form_in_word_rule_for_replace1 in form_in_sentence_rule_for_replace1 and form_in_word_rule_for_replace1 != form_in_sentence_rule_for_replace1:
        start_index_at_form_in_sentence_rule_for_replace1 = form_in_sentence_rule_for_replace1.index(form_in_word_rule_for_replace1)
        end_index_at_form_in_sentence_rule_for_replace1 = start_index_at_form_in_sentence_rule_for_replace1 + len(form_in_word_rule_for_replace1)
        
        before_match = form_in_sentence_rule_for_replace1[:start_index_at_form_in_sentence_rule_for_replace1]
        match = form_in_sentence_rule_for_replace1[start_index_at_form_in_sentence_rule_for_replace1:end_index_at_form_in_sentence_rule_for_replace1]
        after_match = form_in_sentence_rule_for_replace1[end_index_at_form_in_sentence_rule_for_replace1:]
        
        split_form_in_sentence_rule_for_replace1 = [part for part in (before_match, match, after_match) if part]
        
        match_index_at_sentence_rule = split_form_in_sentence_rule_for_replace1.index(match)

        split_form_sentence_rule_for_replace1 = (word_sem_in_sentence_sem_and_index[1][0], split_form_in_sentence_rule_for_replace1, match_index_at_sentence_rule)
        
        return split_form_sentence_rule_for_replace1
    else:
        return []

def can_replace_condition_ability(word_rule_for_replace1, sentence_rule_for_replace1):
    sem_in_word_rule_for_replace1 = word_rule_for_replace1[1]
    sem_in_sentence_rule_for_replace1 = sentence_rule_for_replace1[0]
    
    if sem_in_word_rule_for_replace1 in sem_in_sentence_rule_for_replace1:
        index_at_sem_in_sentence_rule_for_replace1 = sem_in_sentence_rule_for_replace1.index(sem_in_word_rule_for_replace1)
        
        form_in_word_rule_for_replace1 = word_rule_for_replace1[2]
        form_in_sentence_rule_for_replace1 = sentence_rule_for_replace1[1]
        
        if form_in_word_rule_for_replace1 in form_in_sentence_rule_for_replace1 and form_in_word_rule_for_replace1 != form_in_sentence_rule_for_replace1:
            start_index_at_form_in_sentence_rule_for_replace1 = form_in_sentence_rule_for_replace1.index(form_in_word_rule_for_replace1)
            end_index_at_form_in_sentence_rule_for_replace1 = start_index_at_form_in_sentence_rule_for_replace1 + len(form_in_word_rule_for_replace1)
            
            before_match = form_in_sentence_rule_for_replace1[:start_index_at_form_in_sentence_rule_for_replace1]
            match = form_in_sentence_rule_for_replace1[start_index_at_form_in_sentence_rule_for_replace1:end_index_at_form_in_sentence_rule_for_replace1]
            after_match = form_in_sentence_rule_for_replace1[end_index_at_form_in_sentence_rule_for_replace1:]
            
            split_form_in_sentence_rule_for_replace1 = [part for part in (before_match, match, after_match) if part]
            
            match_index_at_sentence_rule = split_form_in_sentence_rule_for_replace1.index(match)

            split_form_sentence_rule_for_replace1 = (word_rule_for_replace1, split_form_in_sentence_rule_for_replace1, match_index_at_sentence_rule)
            
            split_form_sentence_rule_for_replace1 = (sem_in_sentence_rule_for_replace1, split_form_in_sentence_rule_for_replace1)
            return word_rule_for_replace1, split_form_sentence_rule_for_replace1, [1, 0, index_at_sem_in_sentence_rule_for_replace1], [match_index_at_sentence_rule]
        else:
            return []
    else:
        return []

def can_replace_pairs_process(split_set_pair_a_word_rule_and_a_sentence_rule):
    can_replace_pairs = []
    
    for i in split_set_pair_a_word_rule_and_a_sentence_rule:
        word_rule_for_replace1 = i[0]
        sentence_rule_for_replace1 = i[1]
        a_can_replace_pair = can_replace_condition_ability(word_rule_for_replace1, sentence_rule_for_replace1)
        
        if a_can_replace_pair is not None:
            can_replace_pairs.append(a_can_replace_pair)
            
    can_replace_pairs_list = [pair for pair in can_replace_pairs if pair]
    
    return can_replace_pairs_list

def replace_ability(a_can_replace_pair):
    word_rule_for_replace = a_can_replace_pair[0]

    category_label_in_word_rule = a_can_replace_pair[0][0]

    sim_index_list_at_sem_express = a_can_replace_pair[2]

    sim_index_at_sem_express = sim_index_list_at_sem_express[2]

    sim_sem_in_sentence_rule_can_replace = a_can_replace_pair[1][0][sim_index_at_sem_express]

    sim_form_in_sentence_rule_can_replace = a_can_replace_pair[1][1][a_can_replace_pair[3][0]]

    index_to_var = {1: 'p', 2: 'x', 3: 'y'}  # インデックスと変項の対応関係を辞書で定義
    variable_at_sem_express_in_sentence_rule = index_to_var.get(sim_index_at_sem_express, 'x')

    replaced_split_sem_express_in_sentence_rule = list(a_can_replace_pair[1][0])
    replaced_split_sem_express_in_sentence_rule[sim_index_at_sem_express] = variable_at_sem_express_in_sentence_rule

    replaced_split_form_express_in_sentence_rule = list(a_can_replace_pair[1][1])
    sim_index_at_sem_express_in_sentence_rule = a_can_replace_pair[2][2]
    replaced_split_form_express_in_sentence_rule[a_can_replace_pair[3][0]] = f"{category_label_in_word_rule}/{variable_at_sem_express_in_sentence_rule}"
    replaced_form_express_in_sentence_rule = ''.join(replaced_split_form_express_in_sentence_rule)

    replaced_sem_express_in_sentence_rule = (
        f"{replaced_split_sem_express_in_sentence_rule[0]}/_{replaced_split_sem_express_in_sentence_rule[1]}(_{replaced_split_sem_express_in_sentence_rule[2]}, _{replaced_split_sem_express_in_sentence_rule[3]})/{replaced_split_sem_express_in_sentence_rule[4]}"
    )

    replaced_sentence_rule_uncompleted = (replaced_sem_express_in_sentence_rule, "->", replaced_form_express_in_sentence_rule)
    replaced_sentence_rule = ' '.join(replaced_sentence_rule_uncompleted)

    replaced_word_rule = f"{word_rule_for_replace[0]}/_{word_rule_for_replace[1]} -> {word_rule_for_replace[2]}"

    return replaced_sentence_rule, replaced_word_rule

def replace_process(can_replace_pairs):
    replaced_rules = []

    for a_can_replace_pair in can_replace_pairs:
        replaced_sentence_rule, replaced_word_rule = replace_ability(a_can_replace_pair)
        replaced_rules.append(replaced_sentence_rule)
        replaced_rules.append(replaced_word_rule)

    return replaced_rules

def format_replace_pair(replace_pair):
    # replace_pair の2番目の要素を言語知識の形式に変換
    sem_structure, form_strings = replace_pair  # 正しくアンパック
    # sem_structure を適切なフォーマットに変換
    sem_part = f"S/_{sem_structure[1]}(_{sem_structure[2]}, _{sem_structure[3]})/{sem_structure[4]}"
    # form_strings を結合して形式部分を作成
    form_part = "".join(form_strings)
    return f"{sem_part} -> {form_part}"
#  -----------   ここから実行　    --------------------------

only_word_rule_set, only_sentence_rule_set = split_word_rule_set_and_sentence_rule_set(rule_set)
set_index_pair_a_word_rule_and_a_sentence_rule = pairing_index_a_word_rule_and_a_sentence_rule(only_word_rule_set, only_sentence_rule_set)

# セットのペアリングと分割処理を実行
set_pair_a_word_rule_and_a_sentence_rule = set_pair_a_word_rule_and_a_sentence_rule_process(only_word_rule_set, only_sentence_rule_set, set_index_pair_a_word_rule_and_a_sentence_rule)
split_set_pair_a_word_rule_and_a_sentence_rule = split_set_pair_a_word_rule_and_a_sentence_rule_process(set_pair_a_word_rule_and_a_sentence_rule)

# 置換可能なペアの検出
can_replace_pairs = can_replace_pairs_process(split_set_pair_a_word_rule_and_a_sentence_rule)

# ルールの置換
replaced_rules = replace_process(can_replace_pairs)

# ルールセット全体をコピーして保持
remaining_rules = rule_set[:]

# can_replace_pairs の2番目の要素を replacable_pairs に格納
replacable_pairs = []
for replace_pair in can_replace_pairs:
    formatted_rule = format_replace_pair(replace_pair[1])  # 2番目の要素を渡す
    replacable_pairs.append(formatted_rule)

# replacable_pairs を used_rules として扱い、remaining_rules から削除
used_rules = []
for rule in replacable_pairs:
    # 元の rule_set にあるかを確認し、あれば used_rules に追加
    for original_rule in rule_set:
        if rule in original_rule:
            used_rules.append(original_rule)

# 未使用のルールを特定
unapplied_rules = [rule for rule in remaining_rules if rule not in used_rules]

# 統合されたルールと適用できなかったルールを結合
replaced_rules.extend(unapplied_rules)

# 順序を保持しながら重複を削除
replaced_rules = list(dict.fromkeys(replaced_rules))