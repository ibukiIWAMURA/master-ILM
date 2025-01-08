import math
from itertools import combinations
import re
from difflib import SequenceMatcher

def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()  # 前半部分を意味構造 -> .strip()は空白部分を削除
    form = parts[1].strip()  # 後半部分を意味構造
    return semantic_structure, form

def parse_rule_with_cv(rule):
    semantic_structure, form = rule.split('->')
    semantic_parts = semantic_structure.rsplit('/', 1)
    semantic = semantic_parts[0].strip()
    cv = int(semantic_parts[1].strip())
    return semantic, cv, form.strip()

def simply_separate_rule_set_by_cv(rule_set):
    cv_0_rule_set = []
    cv_1_rule_set = []
    for a_rule in rule_set:
        semantic_structure, form = parse_rule(a_rule)
        # スラッシュ以降の値を検出
        deep_structure, cv = semantic_structure.rsplit('/', 1)
        
        if cv == ('0'):
            cv_0_rule_set.append(a_rule)
        else:
            cv_1_rule_set.append(a_rule)
    
    return cv_0_rule_set, cv_1_rule_set

def ibuki_2_the_same_sem_with_different_cv(cv_0_rule_set, cv_1_rule_set):
            
    cv_0_semantics_forms = {parse_rule_with_cv(rule)[0]: rule for rule in cv_0_rule_set}
    cv_1_semantics_forms = {parse_rule_with_cv(rule)[0]: rule for rule in cv_1_rule_set}
    
    ibuki_pairs = []
    for semantic in cv_0_semantics_forms.keys():
        if semantic in cv_1_semantics_forms:
            ibuki_pairs.append((cv_0_semantics_forms[semantic], cv_1_semantics_forms[semantic]))
    
    return ibuki_pairs

def ibuki_form_pairing(ibuki_pairs):
    ibuki_form_pairs = []
    for pair in ibuki_pairs:
        # 各ペアから意味表現部分を抽出
        sem1 = pair[0].split("->")[1].strip()
        sem2 = pair[1].split("->")[1].strip()
        ibuki_form_pairs.append((sem1, sem2))
    return ibuki_form_pairs

def split_form_ability(form1, form2):
    matcher = SequenceMatcher(None, form1, form2)
    split_form1 = []
    split_form2 = []

    # get_opcodes()の出力を確認
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            split_form1.append(form1[i1:i2])
            split_form2.append(form2[j1:j2])
        elif tag == 'replace':
            split_form1.append(form1[i1:i2])
            split_form2.append(form2[j1:j2])
        elif tag == 'delete':
            split_form1.append(form1[i1:i2])
            split_form2.append('')
        elif tag == 'insert':
            split_form1.append('')
            split_form2.append(form2[j1:j2])
    # 空の部分集合を削除
    split_form1 = [part for part in split_form1 if part]
    split_form2 = [part for part in split_form2 if part]
    # print("分割形式", split_form1, split_form2)

    return split_form1, split_form2

def replace_and_return_lists_ability(split_form1, split_form2):
    # 置換マッピング用の辞書
    replace_dict = {}
    next_char = ord('a')  # 置換する文字の開始（アルファベット 'a' から）

    def get_replacement(s):
        """文字列に対応する置換文字を取得し、同じ文字列には同じ置換文字を使用する"""
        nonlocal next_char
        if s not in replace_dict:
            replace_dict[s] = chr(next_char)
            next_char += 1
        return replace_dict[s]

    # list1, list2 の要素をそれぞれ置換
    replaced_form1 = ''.join(get_replacement(item) for item in split_form1)
    replaced_form2 = ''.join(get_replacement(item) for item in split_form2)
    # print("置換形式", replaced_form1, replaced_form2)

    return replaced_form1, replaced_form2

def ibuki_half_normalized_levenshtein_distance_ability(form1, form2):
    inf = float("inf")

    # 文字列の長さに、最初の空白文字の長さを加算する
    s_l = len(form1) + 1
    t_l = len(form2) + 1

    # テーブルを作成
    dp = [[inf] * s_l for _ in range(t_l)]

    # 1行目を埋める
    dp[0] = [i for i in range(s_l)]

    # 1列目を埋める
    for j in range(t_l):
        dp[j][0] = j

    # 2行2列目以降を埋める
    for i in range(1, t_l):
        for j in range(1, s_l):
            left = dp[i][j - 1] + 1
            upp = dp[i - 1][j] + 1
            if form1[j - 1] == form2[i - 1]:
                left_upp = dp[i - 1][j - 1]
            else:
                left_upp = dp[i - 1][j - 1] + 1

            dp[i][j] = min(left, upp, left_upp)

    # 編集距離を返す
    levenshtein_distance =  dp[t_l - 1][s_l - 1]
    # print("普通レーベン",form1, form2, levenshtein_distance)
    complete_normalized_levenshtein_distance = (levenshtein_distance)/(max(len(form1), len(form2)) * 1.00)
    # print("正規化", complete_normalized_levenshtein_distance)
    normalized_levenshtein_distance = complete_normalized_levenshtein_distance / 2
    # print("正規化半分", normalized_levenshtein_distance)
    
    return normalized_levenshtein_distance

def ibuki_form_distance_process(ibuki_form_pairs):
    total_distance = 0
    pair_count = 0
    
    for form1, form2 in ibuki_form_pairs:
        split_form1, split_form2 = split_form_ability(form1, form2)
        replaced_form1, replaced_form2 = replace_and_return_lists_ability(split_form1, split_form2)
        normalized_levenshtein_distance = ibuki_half_normalized_levenshtein_distance_ability(replaced_form1, replaced_form2)
        total_distance += normalized_levenshtein_distance
        pair_count += 1
        

    
    # 平均編集距離を計算
    average_levenshtein_distance = total_distance / pair_count if pair_count > 0 else 0

    return total_distance, average_levenshtein_distance, pair_count

def variance_of_levenshtein_distance_process(ibuki_form_pairs):
    
    # 平均を計算
    total_distance, average_distance, pair_count = ibuki_form_distance_process(ibuki_form_pairs)
    
    # 分散の計算
    total_variance = 0
    pair_count = 0
    
    for form1, form2 in ibuki_form_pairs:
        split_form1, split_form2 = split_form_ability(form1, form2)
        replaced_form1, replaced_form2 = replace_and_return_lists_ability(split_form1, split_form2)
        normalized_levenshtein_distance = ibuki_half_normalized_levenshtein_distance_ability(replaced_form1, replaced_form2)
        variance = (normalized_levenshtein_distance - average_distance) ** 2
        total_variance += variance
        pair_count += 1
    

    # 分散を返す
    variance_of_levenshtein_distance = total_variance / pair_count if pair_count > 0 else 0

    return variance_of_levenshtein_distance, average_distance, pair_count

def Ibuki_4_TopSim(rule_set):
    cv_0_rule_set, cv_1_rule_set = simply_separate_rule_set_by_cv(rule_set)
    ibuki_pairs = ibuki_2_the_same_sem_with_different_cv(cv_0_rule_set, cv_1_rule_set)
    ibuki_form_pairs = ibuki_form_pairing(ibuki_pairs)
    total_distance, average_levenshtein_distance, pair_count = ibuki_form_distance_process(ibuki_form_pairs)
    Ibuki_4_TopSim_variance_of_levenshtein_distance, Ibuki_4_TopSim_average_distance, pair_count = variance_of_levenshtein_distance_process(ibuki_form_pairs)
    
    return Ibuki_4_TopSim_variance_of_levenshtein_distance, Ibuki_4_TopSim_average_distance