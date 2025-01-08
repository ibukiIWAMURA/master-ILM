import re
import itertools
from difflib import SequenceMatcher
import string
# import random
from randomness.Shared_rng import shared_rng


def parse_rule(rule):
    if '->' not in rule:
        return None, None  # 形式表現がない場合に警告
    parts = rule.split('->')
    semantic_structure = parts[0].strip()
    form = parts[1].strip()
    return semantic_structure, form

def clustering_rule_set(rule_set):
    word_rule_set = []
    sentence_rule_set = []
    for rule in rule_set:
        semantic_structure, form = parse_rule(rule)
        if semantic_structure is None or form is None:
            continue
        if semantic_structure.startswith("S/"):
            sentence_rule_set.append(rule)
        else:
            word_rule_set.append(rule)
    return word_rule_set, sentence_rule_set

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

def count_sem_difference_ability(split_sem1, split_sem2):
    differences = 0
    # 2つのリストのうち短い方の長さに合わせてループを回す
    min_length = min(len(split_sem1), len(split_sem2))
    for i in range(min_length): 
        if split_sem1[i] != split_sem2[i]:
            differences += 1
    # もしリストの長さが異なる場合、その分も差異としてカウントする
    differences += abs(len(split_sem1) - len(split_sem2))
    return differences

def count_sem_difference_process(split_semantic_elements_set):
    pairs_with_differences = []
    
    split_sem_1_2_pairs = list(itertools.combinations(split_semantic_elements_set, 2))
    for split_sem1, split_sem2 in split_sem_1_2_pairs:
        differences = count_sem_difference_ability(split_sem1, split_sem2)
        pairs_with_differences.append((split_sem1, split_sem2, differences))
    return pairs_with_differences

def detect_sem_pairs_with_only_one_difference(split_semantic_elements_set):
    detect_sem_pairs_with_only_one_difference = []
    
    split_sem_1_2_pairs = list(itertools.combinations(split_semantic_elements_set, 2))
    for split_sem1, split_sem2 in split_sem_1_2_pairs:
        differences = count_sem_difference_ability(split_sem1, split_sem2)
        if differences == 1:
            detect_sem_pairs_with_only_one_difference.append((split_sem1, split_sem2, differences))
    return detect_sem_pairs_with_only_one_difference

def transformed_set_form(can_chunk_semantic_form_pairs, rule_set):
    rule_dict = {}
    for a_rule in rule_set:
        key, value = a_rule.split("->")
        rule_dict[key] = value

    transformed_form_pairs = []
    for left, right, _ in can_chunk_semantic_form_pairs:
        left_form = f"{left[0]}/{left[1]}({left[2]},{left[3]}){left[4]}"
        right_form = f"{right[0]}/{right[1]}({right[2]},{right[3]}){right[4]}"
        
        # 変換された文字列から記号列を取得
        left_transformed_form = rule_dict.get(left_form, "")
        right_transformed_form = rule_dict.get(right_form, "")

        if left_transformed_form and right_transformed_form:
            transformed_form_pairs.append((left_transformed_form, right_transformed_form))

    return transformed_form_pairs


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
    
    return split_form1, split_form2

def split_form_process(transformed_form_pairs):
    split_form_pairs = []
    for a_form1, a_form2 in transformed_form_pairs:
        a_split_form_result = split_form_sim_diff_ability(a_form1, a_form2)
        split_form_pairs.append(a_split_form_result)
    return split_form_pairs

def compare_forms_by_index_ability(a_form1_as_list, a_form2_as_list): 
    if len(a_form1_as_list) != len(a_form2_as_list):
        return "2"
    
    comparison_form_result_by_index = []
    
    for index in range(len(a_form1_as_list)):
        if a_form1_as_list[index] == a_form2_as_list[index]:
            comparison_form_result_by_index.append('0')
        else:
            comparison_form_result_by_index.append('1')
            
    return ''.join(comparison_form_result_by_index)

def compare_forms_by_index_process(split_form_pairs):
    compare_form_pair_results = []
    for a_form1_as_list, a_form2_as_list in split_form_pairs:
        a_compare_form_pair_result = compare_forms_by_index_ability(a_form1_as_list, a_form2_as_list)
        # compare_form_pair_results.append(a_compare_form_pair_result)
        compare_form_pair_results.append((a_compare_form_pair_result, a_form1_as_list, a_form2_as_list))  # 各ペアを結果と一緒に保存
    return compare_form_pair_results

def contains_uppercase_greek_or_latin(character):
    # ラテン文字の大文字とギリシア文字の大文字に対するチェック
    return character.isupper() or ('Α' <= character <= 'Ω')

def check_for_uppercase_warning(compare_form_pair_results):
    for idx, (result, form1_list, form2_list) in enumerate(compare_form_pair_results):
        # print(f"デバッグ: 現在のインデックス: {idx}, result: {result}, form1_list: {form1_list}, form2_list: {form2_list}")

        for i, char in enumerate(result):
            if char == '1':  # 差異部分を検出
                # print(f"  差異検出: インデックス {i}, form1: {form1_list[i]}, form2: {form2_list[i]}")

                # form1_list[i] と form2_list[i] に大文字のアルファベットまたはギリシア文字が「2つ」含まれているか確認
                uppercase_count = sum(contains_uppercase_greek_or_latin(c) for c in form1_list[i]) + \
                                  sum(contains_uppercase_greek_or_latin(c) for c in form2_list[i])
                # print(f"    大文字のカウント: {uppercase_count}")
                
                if uppercase_count >= 2:  # 2つ以上の大文字が含まれている場合
                    # print(f"Warning: Two or more uppercase letters or Greek letters found at index {i} in form pair.")
                    # print(f"Form1: {form1_list[i]}, Form2: {form2_list[i]}")
                    
                    # ここで result を '2' に変更
                    # print(f"    警告: 2つ以上の大文字が含まれています。result を '2' に変更します。")
                    compare_form_pair_results[idx] = ('2', form1_list, form2_list)

                    
def preprocess_sequence(sequence):
    if not sequence:
        return sequence
    # 最初の要素をリストに追加し、以降は連続する要素を除外
    processed_sequence = [sequence[0]]
    for i in range(1, len(sequence)):
        if not (sequence[i] == '0' and sequence[i - 1] == '0'):
            processed_sequence.append(sequence[i])
    return processed_sequence

def get_filtered_indices(compare_form_pair_results): # chunk可能なペアの　indexを所得
    filtered_indices_set = []
    for index, element in enumerate(compare_form_pair_results):
        # print(f"デバッグ: フィルタリング中のインデックス: {index}, 元の要素: {element}")
        # 前処理を実行
        processed_element = preprocess_sequence(element)
        # print(f"デバッグ: 前処理後の要素: {processed_element}")
        if (
            2 <= len(processed_element) <= 3 and
            processed_element.count('1') < 2 and
            processed_element != '2' and
            not all(c == '0' for c in processed_element)  # すべてが0である場合は対象外
        ):
            filtered_indices_set.append(index)
    # print(f"デバッグ: フィルタリング結果: {filtered_indices_set}")
    return filtered_indices_set

def final_transformed_set_form(can_chunk_semantic_form_pairs, rule_set, filtered_indices_set):
    rule_dict = {}
    for a_rule in rule_set:
        key, value = a_rule.split("->")
        rule_dict[key] = value

    all_transformed_form_pairs = []
    for index in filtered_indices_set:
        transformed_form_pairs = []
        selected_pair = can_chunk_semantic_form_pairs[index]

        left, right, _ = selected_pair
        left_form = f"{left[0]}/{left[1]}({left[2]},{left[3]}){left[4]}"
        right_form = f"{right[0]}/{right[1]}({right[2]},{right[3]}){right[4]}"

        left_transformed_form = rule_dict.get(left_form, "")
        right_transformed_form = rule_dict.get(right_form, "")

        if left_transformed_form:
            transformed_form_pairs.append(f"{left_form}->{left_transformed_form}")
        if right_transformed_form:
            transformed_form_pairs.append(f"{right_form}->{right_transformed_form}")

        all_transformed_form_pairs.append(transformed_form_pairs)
        # print('all trans')
        # print(all_transformed_form_pairs)

    return all_transformed_form_pairs

def transform_only_sem_chunk_pair(can_chunk_rule_set):
    transform_only_sem_chunk_pair_sets = []

    for a_can_chunk_rule in can_chunk_rule_set:
        transformed_a_can_chunk_rule_pair = []
        for an_element_of_a_can_chunk_rule in a_can_chunk_rule:
            semantic_structure, _ = parse_rule(an_element_of_a_can_chunk_rule)
            transformed_a_can_chunk_rule_pair.append(semantic_structure)
        transform_only_sem_chunk_pair_sets.append(transformed_a_can_chunk_rule_pair)
    
    return transform_only_sem_chunk_pair_sets


def split_sem_pairs_for_chunk(sem_chunk_pair_sets):
    split_sem_pairs = []
    for pair in sem_chunk_pair_sets:
        split_pair = []
        for semantic_element in pair:
            split_element = split_semantics_ability(semantic_element)
            split_pair.append(split_element)
        split_sem_pairs.append(split_pair)
    return split_sem_pairs

def detect_index_sem_difference_ability(split_sem1, split_sem2):
    differing_indices = []
    for i in range(len(split_sem1)): 
        if split_sem1[i] != split_sem2[i]:
            differing_indices.append(i)
    return differing_indices

def detect_index_sem_difference_process(split_sem_pairs):
    index_sem_difference_sets = []
    for pair in split_sem_pairs:
        if len(pair) < 2:
            continue  # もしペアが2つの要素を持たない場合、スキップする
        differing_indices = detect_index_sem_difference_ability(pair[0], pair[1])
        index_sem_difference_sets.append(differing_indices)
    return index_sem_difference_sets

def transform_only_form_chunk_pair(can_chunk_rule_set):
    transform_only_form_chunk_pair_sets = []

    for a_can_chunk_rule in can_chunk_rule_set:
        transformed_a_can_chunk_rule_pair = []
        for an_element_of_a_can_chunk_rule in a_can_chunk_rule:
            _, form = parse_rule(an_element_of_a_can_chunk_rule)
            transformed_a_can_chunk_rule_pair.append(form)
        transform_only_form_chunk_pair_sets.append(transformed_a_can_chunk_rule_pair)
    
    return transform_only_form_chunk_pair_sets

def split_form_process(form_chunk_pair_sets):
    split_form_pairs = []
    for pair in form_chunk_pair_sets:
        if len(pair) != 2:
            continue  # 要素数が2でない場合はスキップ
        a_form1, a_form2 = pair
        a_split_form_result = split_form_sim_diff_ability(a_form1, a_form2)
        split_form_pairs.append(a_split_form_result)
    return split_form_pairs

def detect_index_form_difference_ability(a_form1_as_list, a_form2_as_list): 
    if len(a_form1_as_list) != len(a_form2_as_list):
        return "長さが異なります"
    
    differing_indices = []
    
    for index in range(len(a_form1_as_list)):
        if a_form1_as_list[index] != a_form2_as_list[index]:
            differing_indices.append(index)
    
    return differing_indices

def detect_index_form_difference_process(split_form_pairs):
    index_form_difference_sets = []
    for a_form1_as_list, a_form2_as_list in split_form_pairs:
        differing_indices = detect_index_form_difference_ability(a_form1_as_list, a_form2_as_list)
        # print(f"a_form1_as_list: {a_form1_as_list}")
        # print(f"a_form2_as_list: {a_form2_as_list}")
        # print(f"differing_indices: {differing_indices}")
        index_form_difference_sets.append(differing_indices)
    
    return index_form_difference_sets

def used_labels_in_word_rule_set(word_rule_set):
    greek_uppercase = [
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
        'Ν', 'Ξ', 'Ο', 'Π', 'Σ', 'Τ', 'Φ', 'Ψ', 'Ω'
    ]
    
    # 英語のアルファベット大文字とギリシャ文字の大文字を組み合わせる
    all_labels = list(string.ascii_uppercase) + greek_uppercase
    
    # 除外する文字
    excluded_labels = {'S', 'X', 'Y', 'P'}
    
    used_labels = set()
    
    for a_word_rule in word_rule_set:
        for char in a_word_rule:
            if char in all_labels:
                used_labels.add(char)
    # デバッグ: 抽出された used_labels を表示
    # print(f"Used labels in word_rule_set: {used_labels}")
    
    return used_labels

def generate_random_label(used_labels): # どこで used_labelsを取得してる？
    # 英字の大文字とギリシャ文字の大文字を定義
    greek_uppercase = [
        'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
        'Ν', 'Ξ', 'Ο', 'Π', 'Σ', 'Τ', 'Φ', 'Ψ', 'Ω'
    ]
    
    # 英語のアルファベット大文字とギリシャ文字の大文字を組み合わせる
    all_labels = list(string.ascii_uppercase) + greek_uppercase
    
    # 除外する文字
    excluded_labels = {'S', 'X', 'Y', 'P'}
    
    # 使用可能なラベルセットを計算
    available_labels = list(set(all_labels) - used_labels - excluded_labels)
    # print(f"Available labels before selection: {available_labels}")
    # print(f"Currently used labels: {used_labels}")
    
    # ランダムに1つのラベルを選択
    label = shared_rng.choice(available_labels)
    
    # 使用済みのラベルとして追加
    used_labels.add(label)
    
    return label


    
def apply_existing_labels_for_type2_chunk(used_labels, split_form_pairs):
    for pair_index, split_form_pair in enumerate(split_form_pairs):
        # print(f"Processing split_form_pair[{pair_index}]: {split_form_pair}")
        label_count = {}
        for split_form in split_form_pair:
            for element in split_form:
                # print(f"Checking element: {element}")
                for char in element:
                    if contains_uppercase_greek_or_latin(char):  # 大文字のアルファベットまたはギリシャ文字を検出
                        label_count[char] = label_count.get(char, 0) + 1
        
        # 各ラベルのカウント状況を表示
        # print(f"Label count: {label_count}")
        
        for label, count in label_count.items():
            if count >= 2:
                used_labels.add(label)
                # print(f"Added label to used_labels: {label}")
    
    # 最終的な used_labels を表示
    # print(f"Final used_labels: {used_labels}")

def chunk_completed_to_generate_scheme_rules_and_word_rules(
    split_sem_pairs, 
    split_form_pairs, 
    index_sem_difference_sets, 
    index_form_difference_sets, 
    word_rule_set
):
    chunk_completed_pairs = []
    used_labels = set()
    
    used_labels.update(used_labels_in_word_rule_set(word_rule_set))
    apply_existing_labels_for_type2_chunk(used_labels, split_form_pairs)
    
    index_to_var = {1: '_p', 2: '_x', 3: '_y'}
    excluded_labels = {'S', 'X', 'Y', 'P'}  # 必要に応じて、除外したい他のラベルもここに追加
    for sem_pair, form_pair, sem_diff, form_diff in zip(
        split_sem_pairs,
        split_form_pairs,
        index_sem_difference_sets,
        index_form_difference_sets
    ):
        # 現在のルールに含まれるラベルを抽出
        current_labels = set()
        for form_elements in form_pair:
            for element in form_elements:
                for char in element:
                    if contains_uppercase_greek_or_latin(char):
                        current_labels.add(char)
        
        # 使用可能なラベルの絞り込み
        available_labels = set(string.ascii_uppercase + 'ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΣΤΦΨΩ') - used_labels - current_labels - excluded_labels
        if not available_labels:
            raise ValueError("使用可能なラベルがありません！")

        # 既存のラベルを探すロジック
        existing_label = None
        if form_diff:
            for form_elements in form_pair:
                element = form_elements[form_diff[0]]
                for char in element:
                    if contains_uppercase_greek_or_latin(char) and char in current_labels:
                        existing_label = char
                        break
                if existing_label is not None:
                    break
        
        # 既存ラベルが見つかった場合、それを使用し、見つからなければ新たに生成
        if existing_label is not None:
            label = existing_label
        else:
            label = shared_rng.choice(list(available_labels))
            used_labels.add(label)
        
        # 適切な変数名を選択
        index = sem_diff[0]
        var = index_to_var.get(index, '_x')  # '_x'はデフォルト値

        # スキーマルールの作成
        sem_of_scheme_rule = sem_pair[0][:]
        # print("スキーマ作成の意味材料", sem_of_scheme_rule)
        sem_of_scheme_rule[index] = var
        # print("スキーマ作成の変項", sem_of_scheme_rule[index])

        form_of_scheme_rule = form_pair[0][:]  # コピーして変更を加える
        # print("スキーマ作成の形式材料", form_of_scheme_rule)
        
        if form_diff[0] < len(form_of_scheme_rule):
            form_of_scheme_rule[form_diff[0]] = f'{label}/{var[1]}'
            # print("どのラベル付与するの？", label)
            # print("どの変項使うの？", var[1])
        else:
            continue
        
        scheme_rule = [sem_of_scheme_rule, form_of_scheme_rule]
        # print('完成したスキーマ', scheme_rule)

        # 単語ルールを生成
        sem_of_word_rule_1 = [f'{label}', sem_pair[0][index]]
        sem_of_word_rule_2 = [f'{label}', sem_pair[1][index]]
        
        form_of_word_rule_1 = [form_pair[0][form_diff[0]]]
        form_of_word_rule_2 = [form_pair[1][form_diff[0]]]

        word_rule_1 = [sem_of_word_rule_1, form_of_word_rule_1]
        word_rule_2 = [sem_of_word_rule_2, form_of_word_rule_2]

        # 単語ルールとして不要な変数を除外
        word_rules = []
        unwanted_vars = ['_p', '_x', '_y']
        for word_rule in [word_rule_1, word_rule_2]:
            if not any(var in word_rule[0] for var in unwanted_vars):
                word_rules.append(word_rule)

        # 完成したスキーマルールと単語ルールをリストに追加
        chunk_completed_pairs.append((scheme_rule, *word_rules))

    return chunk_completed_pairs

def chunk_learning(rule_set):
    # 1. ルールをクラスタリングして `word_rule_set` を生成し、 `rule_set` からワードルールを削除
    word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
    cv_0_rule_set, cv_1_rule_set = simply_separate_rule_set_by_cv(sentence_rule_set)

    # 2. セマンティクスの分割と差異の検出
    semantic_set_0 = set_semantics(cv_0_rule_set)
    semantic_set_1 = set_semantics(cv_1_rule_set)
    split_semantic_elements_set_0 = split_semantics_process(semantic_set_0)
    split_semantic_elements_set_1 = split_semantics_process(semantic_set_1)
    pairs_with_differences_0 = count_sem_difference_process(split_semantic_elements_set_0)
    pairs_with_differences_1 = count_sem_difference_process(split_semantic_elements_set_1)
    detect_sem_pairs_with_only_one_difference_0 = detect_sem_pairs_with_only_one_difference(split_semantic_elements_set_0)
    detect_sem_pairs_with_only_one_difference_1 = detect_sem_pairs_with_only_one_difference(split_semantic_elements_set_1)

    # 3. フォームの変換
    transformed_form_pairs_0 = transformed_set_form(detect_sem_pairs_with_only_one_difference_0, cv_0_rule_set)
    transformed_form_pairs_1 = transformed_set_form(detect_sem_pairs_with_only_one_difference_1, cv_1_rule_set)
    split_form_pairs_0 = split_form_process(transformed_form_pairs_0)
    split_form_pairs_1 = split_form_process(transformed_form_pairs_1)
    compare_form_pair_results_0 = compare_forms_by_index_process(split_form_pairs_0)
    compare_form_pair_results_1 = compare_forms_by_index_process(split_form_pairs_1)

    # 4. インデックスフィルタリングとルールの最終変換
    check_for_uppercase_warning(compare_form_pair_results_0) # ---------------------------------------------
    filtered_indices_set_0 = get_filtered_indices([result for result, _, _ in compare_form_pair_results_0])
    check_for_uppercase_warning(compare_form_pair_results_1) # ---------------------------------------------
    filtered_indices_set_1 = get_filtered_indices([result for result, _, _ in compare_form_pair_results_1])
    all_transformed_form_pairs_0 = final_transformed_set_form(detect_sem_pairs_with_only_one_difference_0, cv_0_rule_set, filtered_indices_set_0)
    all_transformed_form_pairs_1 = final_transformed_set_form(detect_sem_pairs_with_only_one_difference_1, cv_1_rule_set, filtered_indices_set_1)

    # 5. セマンティクスチャンクとフォームチャンクの処理
    transform_only_sem_chunk_pair_sets_0 = transform_only_sem_chunk_pair(all_transformed_form_pairs_0)
    transform_only_sem_chunk_pair_sets_1 = transform_only_sem_chunk_pair(all_transformed_form_pairs_1)
    split_sem_pairs_0 = split_sem_pairs_for_chunk(transform_only_sem_chunk_pair_sets_0)
    split_sem_pairs_1 = split_sem_pairs_for_chunk(transform_only_sem_chunk_pair_sets_1)
    index_sem_difference_sets_0 = detect_index_sem_difference_process(split_sem_pairs_0)
    index_sem_difference_sets_1 = detect_index_sem_difference_process(split_sem_pairs_1)

    transform_only_form_chunk_pair_sets_0 = transform_only_form_chunk_pair(all_transformed_form_pairs_0)
    transform_only_form_chunk_pair_sets_1 = transform_only_form_chunk_pair(all_transformed_form_pairs_1)
    split_form_pairs_0 = split_form_process(transform_only_form_chunk_pair_sets_0)
    split_form_pairs_1 = split_form_process(transform_only_form_chunk_pair_sets_1)
    index_form_difference_sets_0 = detect_index_form_difference_process(split_form_pairs_0)
    index_form_difference_sets_1 = detect_index_form_difference_process(split_form_pairs_1)

    # 6. チャンクの完了とスキーマルール、ワードルールの生成
    used_labels = used_labels_in_word_rule_set(word_rule_set)
    
    chunk_completed_pairs_0 = chunk_completed_to_generate_scheme_rules_and_word_rules(
    split_sem_pairs_0, 
    split_form_pairs_0, 
    index_sem_difference_sets_0, 
    index_form_difference_sets_0, 
    word_rule_set
    )
    chunk_completed_pairs_1 = chunk_completed_to_generate_scheme_rules_and_word_rules(
    split_sem_pairs_1, 
    split_form_pairs_1, 
    index_sem_difference_sets_1, 
    index_form_difference_sets_1, 
    word_rule_set
    )
    
    # チャンク適用回数をカウントする変数を追加
    chunk_applications_0 = len(all_transformed_form_pairs_0)
    chunk_applications_1 = len(all_transformed_form_pairs_1)
    chunk_applications = chunk_applications_0 + chunk_applications_1
    
    
# ------------------            ここまでやった　　　　　　　　ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー    
    
    
    # ルールセット全体をコピーして保持
    remaining_rules_0 = cv_0_rule_set[:]
    remaining_rules_1 = cv_1_rule_set[:]


    # can_chunk_rule_set に含まれるルールを used_rules に追加
    used_rules_0 = []
    used_rules_1 = []
    
    for rule_pair_0 in all_transformed_form_pairs_0:
        # print(rule_pair_0)
        for rule_0 in rule_pair_0:
            used_rules_0.append(rule_0)
    
    for rule_pair_1 in all_transformed_form_pairs_1:
        for rule_1 in rule_pair_1:
            used_rules_1.append(rule_1)


    # 未使用のルールを特定
    unapplied_rules_0 = [rule for rule in remaining_rules_0 if rule not in used_rules_0]
    unapplied_rules_1 = [rule for rule in remaining_rules_1 if rule not in used_rules_1]

    chunked_rules = []
    for scheme_rule, *word_rules in chunk_completed_pairs_0:
            # スキーマルールの整形
            sem_scheme_rule = scheme_rule[0]
            
            # 2番目と3番目の要素をまとめて括弧で囲む
            if len(sem_scheme_rule) >= 4:
                combined_element = f"({sem_scheme_rule[2]},{sem_scheme_rule[3]})"
                sem_scheme_rule = sem_scheme_rule[:2] + [combined_element] + sem_scheme_rule[4:]

            # スキーマルールを文字列に結合
            sem_scheme_rule = f"{sem_scheme_rule[0]}/" + "".join(sem_scheme_rule[1:])
            form_scheme_rule = "".join(scheme_rule[1])

            chunked_rules.append(f"{sem_scheme_rule}->{form_scheme_rule}")
            
            # 単語ルールの整形
            for word_rule in word_rules:
                sem_word_rule = "/".join(word_rule[0])
                form_word_rule = "".join(word_rule[1])
                chunked_rules.append(f"{sem_word_rule}->{form_word_rule}")
                
    for scheme_rule, *word_rules in chunk_completed_pairs_1:
            # スキーマルールの整形
            sem_scheme_rule = scheme_rule[0]
            
            # 2番目と3番目の要素をまとめて括弧で囲む
            if len(sem_scheme_rule) >= 4:
                combined_element = f"({sem_scheme_rule[2]},{sem_scheme_rule[3]})"
                sem_scheme_rule = sem_scheme_rule[:2] + [combined_element] + sem_scheme_rule[4:]

            # スキーマルールを文字列に結合
            sem_scheme_rule = f"{sem_scheme_rule[0]}/" + "".join(sem_scheme_rule[1:])
            form_scheme_rule = "".join(scheme_rule[1])

            chunked_rules.append(f"{sem_scheme_rule}->{form_scheme_rule}")
            
            # 単語ルールの整形
            for word_rule in word_rules:
                sem_word_rule = "/".join(word_rule[0])
                form_word_rule = "".join(word_rule[1])
                chunked_rules.append(f"{sem_word_rule}->{form_word_rule}")


    chunked_rules.extend(unapplied_rules_0 + unapplied_rules_1)
    chunked_rules.extend(word_rule_set)
    # print("チャンク！",chunked_rules)
    return chunked_rules, chunk_applications