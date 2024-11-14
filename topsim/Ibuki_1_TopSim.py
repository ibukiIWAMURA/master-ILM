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
    total_hamming_distance = 0  # 合計を初期化
    num_elements = len(split_semantic_elements_set)

    # 全てのペアを作成
    pairs = combinations(split_semantic_elements_set, 2)

    # 各ペアに対して Hamming Distance を計算して合計を求める
    pair_count = 0
    for sem1, sem2 in pairs:
        distance = count_hamming_distance_ability(sem1, sem2)
        # print(f"'{sem1}' と '{sem2}' の Hamming distance は {distance}")
        total_hamming_distance += distance  # 合計に加算
        pair_count += 1
    # print(pair_count)
    average_hamming_distance = total_hamming_distance / pair_count

    return average_hamming_distance

def variance_of_hamming_distance_process(split_semantic_elements_set):
    # 平均 Hamming Distance とペアの総数を取得
    average_hamming_distance = count_hamming_distance_process(split_semantic_elements_set)

    total_variance = 0  # 分散の総和
    pair_count = 0  # ペア数のカウント
    pairs = combinations(split_semantic_elements_set, 2)

    # 各ペアに対して (distance - average_hamming_distance)² を計算
    for sem1, sem2 in pairs:
        distance = count_hamming_distance_ability(sem1, sem2)
        variance = (distance - average_hamming_distance) ** 2
        total_variance += variance
        pair_count += 1  # ペア数をカウント
    variance_of_hamming_distance = total_variance / pair_count

    return variance_of_hamming_distance

def standard_deviation_of_hamming_distance_process(variance_of_hamming_distance):
    return math.sqrt(variance_of_hamming_distance)

def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()  # 前半部分を意味構造 -> .strip()は空白部分を削除
    form = parts[1].strip()  # 後半部分を意味構造
    return semantic_structure, form

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
    levenshtein_distance =  dp[t_l - 1][s_l - 1]
    return levenshtein_distance


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

    return replaced_form1, replaced_form2

def ibuki_form_distance_ability(form1, form2):

    levenshtein_distance = levenshtein_distance_ability(form1, form2)
    split_form1, split_form2 = split_form_ability(form1, form2)
    replaced_form1, replaced_form2 = replace_and_return_lists_ability(split_form1, split_form2)
    replaced_levenshtein_distance = levenshtein_distance_ability(replaced_form1, replaced_form2)
    if levenshtein_distance == replaced_levenshtein_distance:
        ibuki_distance = levenshtein_distance / 2
    else:
        ibuki_distance = levenshtein_distance - replaced_levenshtein_distance
    form_distance = ibuki_distance/(max(len(form1), len(form2)) * 1.00)
    return form_distance


def ibuki_form_distance_process(form_set):
    total_distance = 0
    pair_count = 0
    # 全てのペアを生成
    pairs = combinations(form_set, 2)
    
    for form1, form2 in pairs:
        distance = ibuki_form_distance_ability(form1, form2)
        # print(f"'{form1}' と '{form2}' の Levenshtein distance は {distance}")
        total_distance += distance
        pair_count += 1
    
    # 平均編集距離を計算
    average_levenshtein_distance = total_distance / pair_count if pair_count > 0 else 0

    return total_distance, average_levenshtein_distance, pair_count

def variance_of_levenshtein_distance_process(form_set):
    pairs = list(combinations(form_set, 2))
    # 平均を計算
    total_distance, average_distance, pair_count = ibuki_form_distance_process(form_set)

    # 分散の計算
    total_variance = 0
    for form1, form2 in pairs:
        distance = levenshtein_distance_ability(form1, form2)
        variance = (distance - average_distance) ** 2
        total_variance += variance

    # 分散を返す
    variance_of_levenshtein_distance = total_variance / pair_count if pair_count > 0 else 0

    return variance_of_levenshtein_distance, average_distance, pair_count

def standard_deviation_of_levenshtein_distance_process(variance_of_levenshtein_distance):
    return math.sqrt(variance_of_levenshtein_distance)

def correlation_coefficient(
    split_semantic_elements_set,
    form_set,
    average_hamming_distance,
    average_levenshtein_distance,
    standard_deviation_of_hamming_distance,
    standard_deviation_of_levenshtein_distance,
    pair_count
):

    total_covariance = 0  # 共分散の総和
    pairs_sem = combinations(split_semantic_elements_set, 2)
    pairs_form = combinations(form_set, 2)

    for (sem1, sem2), (form1, form2) in zip(pairs_sem, pairs_form):
        hamming_distance = count_hamming_distance_ability(sem1, sem2)
        levenshtein_distance = ibuki_form_distance_ability(form1, form2)

        # (意味類似度 - 平均) と (形式類似度 - 平均) を掛け合わせて共分散の合計に加算
        total_covariance += (hamming_distance - average_hamming_distance) * (levenshtein_distance - average_levenshtein_distance)

    # 共分散をペア数で割る（共分散の平均を求める）
    covariance = total_covariance / pair_count if pair_count > 0 else 0

    # 相関係数 = 共分散 / (標準偏差_意味 * 標準偏差_形式)
    if standard_deviation_of_hamming_distance > 0 and standard_deviation_of_levenshtein_distance > 0:
        TopSim_value = covariance / (standard_deviation_of_hamming_distance * standard_deviation_of_levenshtein_distance)
    else:
        TopSim_value = 0  # 標準偏差が 0 の場合は相関係数を 0 とする

    return TopSim_value

def Ibuki_1_TopSim(rule_set):

    semantic_set = set_semantics(rule_set)
    split_semantic_elements_set = split_semantics_process(semantic_set)
    average_hamming_distance = count_hamming_distance_process(split_semantic_elements_set)
    variance_of_hamming_distance = variance_of_hamming_distance_process(split_semantic_elements_set)
    standard_deviation_of_hamming_distance = standard_deviation_of_hamming_distance_process(variance_of_hamming_distance)
    form_set = set_form(rule_set)
    total_distance, average_levenshtein_distance, pair_count = ibuki_form_distance_process(form_set)
    variance_of_levenshtein_distance, average_distance, pair_count = variance_of_levenshtein_distance_process(form_set)
    standard_deviation_of_levenshtein_distance = standard_deviation_of_levenshtein_distance_process(variance_of_levenshtein_distance)
    Ibuki_1_TopSim_value = correlation_coefficient(
        split_semantic_elements_set,
        form_set,
        average_hamming_distance,
        average_levenshtein_distance,
        standard_deviation_of_hamming_distance,
        standard_deviation_of_levenshtein_distance,
        pair_count
        )
    return Ibuki_1_TopSim_value