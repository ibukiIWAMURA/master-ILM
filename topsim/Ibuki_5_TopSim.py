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

def split_form_by_index_ability(form):
    # 1文字ずつ分割
    chars = list(form)
    
    # 後処理で [A-ZΑ-Ω]/[xyp] を統合
    splited_form = []
    i = 0
    while i < len(chars):
        if i < len(chars) - 2 and re.match(r'[A-ZΑ-Ω]', chars[i]) and chars[i + 1] == '/' and chars[i + 2] in 'xyp':
            # 大文字/小文字形式を統合
            splited_form.append(''.join(chars[i:i+3]))
            i += 3  # 統合した分だけ進める
        else:
            # そのまま追加
            splited_form.append(chars[i])
            i += 1
    return splited_form

def merge_segmentation_once_ability(splited_form1, splited_form2):
    label_pattern = re.compile(r'[A-ZΑ-Ω]/[xyp]')
    merged_list1, merged_list2 = [], []

    while splited_form1 and splited_form2:
        # デバッグ情報を表示
        # print(f"現在のリスト状態: splited_form1={splited_form1}, splited_form2={splited_form2}")

        # 最長の連続する共通部分を探す
        longest_match_length = 0
        start_index1, start_index2 = -1, -1

        for index1 in range(len(splited_form1)):
            for index2 in range(len(splited_form2)):
                match_length = 0
                while (index1 + match_length < len(splited_form1) and
                       index2 + match_length < len(splited_form2) and
                       splited_form1[index1 + match_length] == splited_form2[index2 + match_length] and
                       not label_pattern.match(splited_form1[index1 + match_length])):  # 結合対象外条件
                    match_length += 1
                if match_length > longest_match_length:
                    longest_match_length = match_length
                    start_index1, start_index2 = index1, index2

        if longest_match_length == 0:
            # 共通部分がない場合、各リストの先頭要素を結果に追加
            # print(f"共通部分が見つかりません: splited_form1={splited_form1}, splited_form2={splited_form2}")
            merged_list1.append(splited_form1.pop(0))
            if splited_form2:
                merged_list2.append(splited_form2.pop(0))
        else:
            # 共通部分が見つかった場合、その部分をマージ
            common_segment = ''.join(splited_form1[start_index1:start_index1 + longest_match_length])
            # print(f"共通部分を発見: {common_segment}")

            # 非共通部分を表示
            pre_common1 = ''.join(splited_form1[:start_index1])  # 共通部分の前
            pre_common2 = ''.join(splited_form2[:start_index2])  # 共通部分の前
            # print(f"非共通部分（共通部分の前）: {pre_common1}, {pre_common2}")

            # 共通部分の後をリスト化して保存
            post_common1 = splited_form1[start_index1 + longest_match_length:]  # 共通部分の後
            post_common2 = splited_form2[start_index2 + longest_match_length:]  # 共通部分の後
            # print(f"非共通部分（共通部分の後）: {post_common1}, {post_common2}")

            # 共通部分の前の部分を結果に追加
            merged_list1.extend(splited_form1[:start_index1])
            merged_list2.extend(splited_form2[:start_index2])

            # マージした共通部分を結果に追加
            merged_list1.append(common_segment)
            merged_list2.append(common_segment)

            # 共通部分以降を元のリストから削除
            splited_form1 = post_common1
            splited_form2 = post_common2

    # 残りの要素を結果に追加
    # print(f"残りの要素を追加: splited_form1={splited_form1}, splited_form2={splited_form2}")
    merged_list1.extend(splited_form1)
    merged_list2.extend(splited_form2)

    return merged_list1, merged_list2

def merge_segmentation_iteration_ability(A, B):
    
    segment_list1 = split_form_by_index_ability(A)
    segment_list2 = split_form_by_index_ability(B)
    
    previous_result1, previous_result2 = None, None
    while previous_result1 != segment_list1 or previous_result2 != segment_list2:
        # 前回の結果を保存して比較
        previous_result1, previous_result2 = segment_list1, segment_list2
        # print(f"マージ処理開始: previous_result1={previous_result1}, previous_result2={previous_result2}")
        segment_list1, segment_list2 = merge_segmentation_once_ability(segment_list1[:], segment_list2[:])
        # print(f"マージ処理結果: segment_list1={segment_list1}, segment_list2={segment_list2}")

    return segment_list1, segment_list2

def group_seg(segment_list1, segment_list2):
    # 共通する要素を取得
    set1, set2 = set(segment_list1), set(segment_list2)
    common_elements = list(set1.intersection(set2))
    
    # 共通しない要素を取得
    unique_to_list1 = [(element, idx) for idx, element in enumerate(segment_list1) if element not in set2]
    # print("なにこれ1", unique_to_list1)
    unique_to_list2 = [(element, idx) for idx, element in enumerate(segment_list2) if element not in set1]
    # print("なにこれ2", unique_to_list2)
    
    
    return common_elements, unique_to_list1, unique_to_list2

def combine_elements_by_index(list_A, indexed_elements):
    
    # print("初期の list_A:", list_A)
    # print("初期の indexed_elements:", indexed_elements)
    # Extract indices from indexed_elements and sort them
    indexed_elements.sort(key=lambda x: x[1])
    # print("ソート後の indexed_elements:", indexed_elements)
    indices = [index for _, index in indexed_elements]
    # print("抽出されたインデックス:", indices)
    
    # Check if indices is empty
    if not indices:
        # print("インデックスが空です。元の list_A を返します。")
        return list_A

    # Group adjacent indices
    grouped_indices = []
    current_group = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] == indices[i - 1] + 1:
            current_group.append(indices[i])
        else:
            grouped_indices.append(current_group)
            current_group = [indices[i]]

    grouped_indices.append(current_group)  # Add the last group
    # print("グループ化されたインデックス:", grouped_indices)

    # Merge elements in list_A based on grouped_indices
    for group in grouped_indices:
        if len(group) > 1:
            merged_value = ''.join(list_A[i] for i in group)
            # print(f"結合された値: {merged_value} (インデックス: {group})")
            list_A[group[0]] = merged_value
            for i in group[1:]:
                list_A[i] = None  # Mark as None to remove later
                # print(f"インデックス {i} の要素を None に設定")

    # print("結合後の list_A (None 含む):", list_A)
    
    # Remove None values from list_A
    final_result = [x for x in list_A if x is not None]
    # print("最終結果:", final_result)

    return final_result

def organize_split_form(split_form_n):
    regex = r'[A-ZΑ-Ω]/[xyp]'
    processed_list = []
    pattern = re.compile(regex)

    for element in split_form_n:
        matches = list(pattern.finditer(element))
        if matches:
            # If there are matches, split the string around them
            last_index = 0
            for match in matches:
                start, end = match.span()
                # Add the part before the match if it exists
                if start > last_index:
                    processed_list.append(element[last_index:start])
                # Add the match itself
                processed_list.append(element[start:end])
                last_index = end
            # Add the remaining part after the last match if it exists
            if last_index < len(element):
                processed_list.append(element[last_index:])
        else:
            # If no match, keep the element as is
            processed_list.append(element)

    return processed_list

def split_form_sim_diff_ability(a_form1, a_form2):
    segment_list1, segment_list2 = merge_segmentation_iteration_ability(a_form1, a_form2)
    common_elements, unique_to_list1, unique_to_list2 = group_seg(segment_list1, segment_list2)

    pre_split_form1 = combine_elements_by_index(segment_list1, unique_to_list1)
    split_form1 = organize_split_form(pre_split_form1)
    pre_split_form2 = combine_elements_by_index(segment_list2, unique_to_list2)
    split_form2 = organize_split_form(pre_split_form2)
    
    # print("分割した形式", split_form1, split_form2)
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

def as_set_ibuki_levenshtein_distance_ability(replaced_form1, replaced_form2):

    # 形式表現を集合として扱う
    set_form1 = set(replaced_form1)
    set_form2 = set(replaced_form2)

    # 対称差を計算
    symmetric_difference = set_form1.symmetric_difference(set_form2)

    # 距離を対称差の要素数として返す
    as_set_ibuki_levenshtein_distance = (len(symmetric_difference) / 2)/(max(len(replaced_form1), len(replaced_form2)) * 2.00)
    return as_set_ibuki_levenshtein_distance

def as_set_ibuki_form_distance_process(ibuki_form_pairs):
    total_distance = 0
    pair_count = 0

        
    for form1, form2 in ibuki_form_pairs:
        split_form1, split_form2 = split_form_sim_diff_ability(form1, form2)
        # print("分割形式", split_form1, split_form2)
        replaced_form1, replaced_form2 = replace_and_return_lists_ability(split_form1, split_form2)
        # print("置換形式", replaced_form1, replaced_form2)
        as_set_ibuki_levenshtein_distance = as_set_ibuki_levenshtein_distance_ability(replaced_form1, replaced_form2)
        # print(f"'{replaced_form1}' と '{replaced_form2}' の Levenshtein distance は {as_set_ibuki_levenshtein_distance}")
        total_distance += as_set_ibuki_levenshtein_distance
        pair_count += 1

    
    # 平均編集距離を計算
    average_levenshtein_distance = total_distance / pair_count if pair_count > 0 else 0

    return total_distance, average_levenshtein_distance, pair_count

def variance_of_levenshtein_distance_process(ibuki_form_pairs):
    
    # 平均を計算
    total_distance, average_distance, pair_count = as_set_ibuki_form_distance_process(ibuki_form_pairs)
    
    # 分散の計算
    total_variance = 0
    pair_count = 0
    
    for form1, form2 in ibuki_form_pairs:
        split_form1, split_form2 = split_form_sim_diff_ability(form1, form2)
        replaced_form1, replaced_form2 = replace_and_return_lists_ability(split_form1, split_form2)
        normalized_levenshtein_distance = as_set_ibuki_levenshtein_distance_ability(replaced_form1, replaced_form2)
        variance = (normalized_levenshtein_distance - average_distance) ** 2
        total_variance += variance
        pair_count += 1

    # 分散を返す
    variance_of_levenshtein_distance = total_variance / pair_count if pair_count > 0 else 0

    return variance_of_levenshtein_distance, average_distance, pair_count

def Ibuki_5_TopSim(rule_set):
    cv_0_rule_set, cv_1_rule_set = simply_separate_rule_set_by_cv(rule_set)
    ibuki_pairs = ibuki_2_the_same_sem_with_different_cv(cv_0_rule_set, cv_1_rule_set)
    ibuki_form_pairs = ibuki_form_pairing(ibuki_pairs)
    total_distance, average_levenshtein_distance, pair_count = as_set_ibuki_form_distance_process(ibuki_form_pairs)
    Ibuki_5_TopSim_variance_of_levenshtein_distance, Ibuki_5_TopSim_average_distance, pair_count = variance_of_levenshtein_distance_process(ibuki_form_pairs)
    
    return Ibuki_5_TopSim_variance_of_levenshtein_distance, Ibuki_5_TopSim_average_distance