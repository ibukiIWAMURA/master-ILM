from scipy.stats import spearmanr
from scipy.stats import pearsonr
import math
from itertools import combinations
import re
from difflib import SequenceMatcher

def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()  # 前半部分を意味構造 -> .strip()は空白部分を削除
    form = parts[1].strip()  # 後半部分を意味構造
    return semantic_structure, form

def set_semantics(rule_set):
    semantic_set = []
    for a_rule in rule_set:
        a_semantics = parse_rule(a_rule)[0]
        semantic_set.append(a_semantics)
    return semantic_set

def split_semantics_ability(semantic_elements):
    # 意味表現を単語単位で分割
    return re.findall(r'_[a-zA-Z0-9]+|\(\w+\)|[A-Z]+|/[0-9]', semantic_elements)

def split_semantics_process(semantic_set):
    split_semantic_elements_set = []
    for a_semantic_element in semantic_set:
        one_of_semantic_set = split_semantics_ability(a_semantic_element)
        split_semantic_elements_set.append(one_of_semantic_set)
    return split_semantic_elements_set

def count_hamming_distance_ability(sem1, sem2):
    """Hamming Distanceの計算"""
    count = 0
    for i in range(len(sem1)):
        if sem1[i] != sem2[i]:
            count += 1
    return count

def count_hamming_distance_process(split_semantic_elements_set):
    
    hamming_distance_list = []
    # 全てのペアを作成
    pairs = combinations(split_semantic_elements_set, 2)
    for sem1, sem2 in pairs:
        distance = count_hamming_distance_ability(sem1, sem2)
        hamming_distance_list.append(distance)
    return hamming_distance_list

def set_form(rule_set):
    form_set = []
    for a_rule in rule_set:
        a_form = parse_rule(a_rule)[1]
        form_set.append(a_form)
    return form_set

def levenshtein_distance_ability(form1, form2):
    """
    2つの文字列 form1 と form2 の Levenshtein distance（編集距離）を計算する関数
    """
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
    return dp[t_l - 1][s_l - 1]

def levenshtein_distance_process(form_set):
    form_distance_list = []
    # 全てのペアを生成
    pairs = combinations(form_set, 2)

    # 各ペアに対して levenshtein_distance_ability を計算
    for form1, form2 in pairs:
        distance = levenshtein_distance_ability(form1, form2)
        form_distance_list.append(distance)
    return form_distance_list


def TopSim(rule_set):
    
    semantic_set = set_semantics(rule_set)
    split_semantic_elements_set = split_semantics_process(semantic_set)
    form_set = set_form(rule_set)
    
    hamming_distance_list = count_hamming_distance_process(split_semantic_elements_set)
    form_distance_list = levenshtein_distance_process(form_set)
    
    TopSim_value, TopSim_p_value = pearsonr(hamming_distance_list, form_distance_list)
    return TopSim_value, TopSim_p_value