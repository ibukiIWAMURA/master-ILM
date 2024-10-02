import random
import re

def set_only_word_rule(rule_set):
    pattern = re.compile(r'^[A-Z]/\w+->\w+$')
    only_word_rule_set = [rule for rule in rule_set if pattern.match(rule)]
    return only_word_rule_set

def word_rule_pair(only_word_rule_set):
    pairs = []
    n = len(only_word_rule_set)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((only_word_rule_set[i], only_word_rule_set[j]))
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

def split_word_rule(word_rule):
    category_label, rest = word_rule.split('/')
    meaning, form = rest.split('->')
    return [category_label, meaning, form]

def category_integration_ability(split_word_rule1, split_word_rule2):
    unified_category_label = random.choice([split_word_rule1[0], split_word_rule2[0]])
    return f"{unified_category_label}/{split_word_rule1[1]}->{split_word_rule1[2]}"

def category_integration_learning(rule_set):
    # 単語ルールを抽出
    only_word_rule_set = set_only_word_rule(rule_set)
    pairs = word_rule_pair(only_word_rule_set)
    integrable_pairs = can_category_integration_pair_set(pairs)
    
    # 統合可能なルールペアから統合ルールを生成
    integrated_rules = []
    for pair in integrable_pairs:
        split_rule1 = split_word_rule(pair[0])
        split_rule2 = split_word_rule(pair[1])
        unified_rule = category_integration_ability(split_rule1, split_rule2)
        integrated_rules.append(unified_rule)
    
    # 統合されなかったルールをunintegrable_rulesとして保持
    integrable_set = set()
    for pair in integrable_pairs:
        integrable_set.add(pair[0])
        integrable_set.add(pair[1])
    
    unintegrable_rules = [rule for rule in rule_set if rule not in integrable_set]
    
    # 統合されたルールと統合されなかったルールを統合して返す
    integrated_rules.extend(unintegrable_rules)
    
    return integrated_rules