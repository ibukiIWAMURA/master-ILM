import re
import itertools
from difflib import SequenceMatcher
import string
import random

# グローバル変数でINDUCTION_ACCURACYを設定
INDUCTION_ACCURACY = 0.8  # デフォルト値

def set_induction_accuracy(value):
    global INDUCTION_ACCURACY
    INDUCTION_ACCURACY = value

# def parse_rule(rule):
#     # ルールに '->' が含まれているか確認
#     if '->' not in rule:
#         print(f"警告: 期待される形式 '->' がないルールが見つかりました: {rule}")
#         return None, None  # または適切なデフォルト値を設定する
    
#     parts = rule.split('->')
#     semantic_structure = parts[0].strip()  # 前半部分を意味構造 -> .strip()は空白部分を削除
#     form = parts[1].strip()  # 後半部分を意味構造
#     print(f"parse_rule - 分解後: semantic_structure='{semantic_structure}', form='{form}'")  # デバッグ出力

#     return semantic_structure, form

def parse_rule(rule):
    if '->' not in rule:
        # print(f"警告: 期待される形式 '->' がないルールが見つかりました: {rule}") # ------------------------------------------------1114
        return None, None  # 形式表現がない場合に警告
    parts = rule.split('->')
    semantic_structure = parts[0].strip()
    form = parts[1].strip()
    # print(f"parse_rule - 分解後: semantic_structure='{semantic_structure}', form='{form}'")  # デバッグ出力 # ------------------------------------------------1114
    return semantic_structure, form

# def clustering_rule_set(rule_set):
#     word_rule_set = []
#     sentence_rule_set = []
#     for rule in rule_set:
#         semantic_structure, _ = parse_rule(rule)
#         if semantic_structure.startswith("S/"):
#             sentence_rule_set.append(rule)
#         else:
#             word_rule_set.append(rule)
#     return word_rule_set, sentence_rule_set

def clustering_rule_set(rule_set):
    word_rule_set = []
    sentence_rule_set = []
    for rule in rule_set:
        semantic_structure, form = parse_rule(rule)
        if semantic_structure is None or form is None:
            # print(f"clustering_rule_set - 無効なルール: {rule}")  # 形式が無効なルール # ------------------------------------------------1114
            continue
        if semantic_structure.startswith("S/"):
            sentence_rule_set.append(rule)
        else:
            word_rule_set.append(rule)
    # print(f"clustering_rule_set - 分割結果: word_rule_set={word_rule_set}, sentence_rule_set={sentence_rule_set}") # ------------------------------------------------1114
    return word_rule_set, sentence_rule_set

# def accurate_induction_probability_of_cv(cv, accurate_induction_probability=INDUCTION_ACCURACY):
#     if random.random() < accurate_induction_probability:
#         return cv # 正しい推論
#     else:
#         return '1' if cv == '0' else '0'  # 誤って反対の値を推論する場合
    
# def accurate_induction_probability_of_cv(cv, accurate_induction_probability=None):
#     # 引数が None の場合はグローバル変数を使う
#     if accurate_induction_probability is None:
#         accurate_induction_probability = INDUCTION_ACCURACY
#     if random.random() < accurate_induction_probability:
#         return cv  # 正しい推論
#     else:
#         return '1' if cv == '0' else '0'  # 誤って反対の値を推論する場合    
    
# CV 推論結果のキャッシュ
cv_cache = {}

def accurate_induction_probability_of_cv(cv, rule, accurate_induction_probability=None):
    global cv_cache  # キャッシュをグローバルに利用
    
    # キャッシュにルールが存在する場合はその値を返す
    if rule in cv_cache:
        return cv_cache[rule]
    
    # 引数が None の場合はグローバル変数を使う
    if accurate_induction_probability is None:
        accurate_induction_probability = INDUCTION_ACCURACY

    # 確率に基づいて推論結果を決定
    inferred_cv = cv if random.random() < accurate_induction_probability else '1' if cv == '0' else '0'
    
    # キャッシュに推論結果を保存
    cv_cache[rule] = inferred_cv
    
    return inferred_cv

    
# def separate_rule_set_by_cv(rule_set):
#     cv_0_rule_set = []
#     cv_1_rule_set = []
    
#     for a_rule in rule_set:
#         semantic_structure, form = parse_rule(a_rule)
#         # スラッシュ以降の値を検出
#         deep_structure, cv = semantic_structure.rsplit('/', 1)
#         induced_cv = accurate_induction_probability_of_cv(cv, a_rule)
        
#         # 推論された値でルールを書き換え
#         a_rule_as_a_result_of_induction = f"{deep_structure}/{induced_cv}->{form}"
#         print(f"separate_rule_set_by_cv - 推論後: {a_rule_as_a_result_of_induction}")
        
        
#         if induced_cv == ('0'):
#             cv_0_rule_set.append(a_rule_as_a_result_of_induction)
#         else:
#             cv_1_rule_set.append(a_rule_as_a_result_of_induction)
    
#     return cv_0_rule_set, cv_1_rule_set

def separate_rule_set_by_cv(rule_set):
    cv_0_rule_set = []
    cv_1_rule_set = []
    for a_rule in rule_set:
        semantic_structure, form = parse_rule(a_rule)
        if semantic_structure is None or form is None:
            # print(f"separate_rule_set_by_cv - 無効なルール: {a_rule}") # ------------------------------------------------1114
            continue
        deep_structure, cv = semantic_structure.rsplit('/', 1)
        induced_cv = accurate_induction_probability_of_cv(cv, a_rule)
        a_rule_as_a_result_of_induction = f"{deep_structure}/{induced_cv}->{form}"
        # print(f"separate_rule_set_by_cv - 推論後: {a_rule_as_a_result_of_induction}") # ------------------------------------------------1114
        if induced_cv == '0':
            cv_0_rule_set.append(a_rule_as_a_result_of_induction)
        else:
            cv_1_rule_set.append(a_rule_as_a_result_of_induction)
    # print(f"separate_rule_set_by_cv - 結果: cv_0_rule_set={cv_0_rule_set}, cv_1_rule_set={cv_1_rule_set}") # ------------------------------------------------1114
    return cv_0_rule_set, cv_1_rule_set

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
    # can_chunk_semantic_form_pairs  = detect_sem_pairs_with_only_one_difference
    # ルールセットを辞書に変換 : 各ルールの左辺をキー、右辺を値とします
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

def preprocess_form(form):
    # 正規表現で「大文字 + スラッシュ + 小文字 (x, y, p)」をひとつの塊として扱う
    pattern = re.compile(r'([A-ZΑ-Ω])/(x|y|p)')
    
    # 見つかった部分を一時的に特殊文字で置き換える
    processed_form = form
    matches = pattern.findall(form)
    replaced_parts = []
    
    for match in matches:
        # 例えば「W/x」を特殊なトークンに置き換える (例: "_XYP0_")
        part = f"{match[0]}/{match[1]}"
        token = f"_XYP{len(replaced_parts)}_"
        processed_form = processed_form.replace(part, token, 1)
        replaced_parts.append((token, part))
    
    return processed_form, replaced_parts

def postprocess_form(processed_form, replaced_parts):
    # 特殊トークンを元のカテゴリーラベルセットに戻す
    for token, original in replaced_parts:
        processed_form = processed_form.replace(token, original)
    return processed_form

 # トークンを含む要素を分割して、それぞれを独立した要素にする
def split_token_parts(split_form):
    final_split_form = []
    for part in split_form:
        # 特殊トークンを含んでいる部分を見つけたら、その前後の部分と分割
        if "_XYP" in part:
            parts = re.split(r'(_XYP\d+_)', part)
            final_split_form.extend([p for p in parts if p])  # 空要素を除外
        else:
            final_split_form.append(part)
    return final_split_form

def split_form_sim_diff_ability(a_form1, a_form2):
    # まずはフォームを前処理して、カテゴリーラベルセットを特殊トークンに置き換える
    processed_form1, replaced_parts1 = preprocess_form(a_form1)
    processed_form2, replaced_parts2 = preprocess_form(a_form2)

    # SequenceMatcher で比較
    matcher = SequenceMatcher(None, processed_form1, processed_form2)
    split_form1 = []
    split_form2 = []

    # get_opcodes()の出力を確認
    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        part1 = processed_form1[i1:i2]
        part2 = processed_form2[j1:j2]

        if tag == 'equal':
            split_form1.append(part1)
            split_form2.append(part2)
        elif tag == 'replace':
            split_form1.append(part1)
            split_form2.append(part2)
        elif tag == 'delete':
            split_form1.append(part1)
            split_form2.append('')
        elif tag == 'insert':
            split_form1.append('')
            split_form2.append(part2)

    # 空の部分集合を削除
    split_form1 = [part for part in split_form1 if part]
    split_form2 = [part for part in split_form2 if part]
    # split_form1 と split_form2 をトークン部分ごとに分割
    split_form1 = split_token_parts(split_form1)
    split_form2 = split_token_parts(split_form2)

    # 結果を元に戻す (トークンを元のカテゴリーラベルセットに復元)
    split_form1 = [postprocess_form(part, replaced_parts1) for part in split_form1]
    split_form2 = [postprocess_form(part, replaced_parts2) for part in split_form2]

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
        for i, char in enumerate(result):
            if char == '1':  # 差異部分を検出
                # form1_list[i] と form2_list[i] に大文字のアルファベットまたはギリシア文字が「2つ」含まれているか確認
                uppercase_count = sum(contains_uppercase_greek_or_latin(c) for c in form1_list[i]) + \
                                  sum(contains_uppercase_greek_or_latin(c) for c in form2_list[i])
                
                if uppercase_count >= 2:  # 2つ以上の大文字が含まれている場合
                    # print(f"Warning: Two or more uppercase letters or Greek letters found at index {i} in form pair.")
                    # print(f"Form1: {form1_list[i]}, Form2: {form2_list[i]}")
                    
                    # ここで result を '2' に変更
                    compare_form_pair_results[idx] = ('2', form1_list, form2_list)

def get_filtered_indices(compare_form_pair_results): # chunk可能なペアの　indexを所得
    filtered_indices_set = []
    for index, element in enumerate(compare_form_pair_results):
        if (
            2 <= len(element) <= 3 and
            element.count('1') < 2 and
            # '00' not in element and # 同じものが並んでいる　    ---------------------------------ここの条件を外した
            # '11' not in element and # 異なるものが並んでいる　   ---------------------------------ここの条件を外した
            element != '2' and
            not all(c == '0' for c in element)  # すべてが0である場合は対象外
        ):
            filtered_indices_set.append(index)
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
    label = random.choice(available_labels)
    
    # 使用済みのラベルとして追加
    used_labels.add(label)
    
    return label

def apply_existing_labels_for_type2_chunk(used_labels, split_form_pairs):
    for split_form_pair in split_form_pairs:
        label_count = {}
        for split_form in split_form_pair:
            for element in split_form:
                if '/' in element:
                    label, _ = element.split('/')
                    label_count[label] = label_count.get(label, 0) + 1
        
        # デバッグ: 各ラベルのカウント状況を表示
        # print(f"Label count in split_form_pair: {label_count}")
        
        for label, count in label_count.items():
            if count >= 2:
                used_labels.add(label)
                
    # デバッグ: 関数終了時の used_labels の状態
    # print(f"Used labels after apply_existing_labels_for_type2_chunk: {used_labels}")

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

def chunk_completed_to_generate_scheme_rules_and_word_rules(
    split_sem_pairs, split_form_pairs, index_sem_difference_sets, 
    index_form_difference_sets, word_rule_set
):
    chunk_completed_pairs = []
    used_labels = set()

    # print('単語ルール:', word_rule_set) # ------------------------------------------------1114
    used_labels.update(used_labels_in_word_rule_set(word_rule_set))
    # print('Initial used_labels:', used_labels) # ------------------------------------------------1114
    
    apply_existing_labels_for_type2_chunk(used_labels, split_form_pairs)
    # print('Used labels after applying existing labels:', used_labels) # ------------------------------------------------1114

    index_to_var = {1: '_p', 2: '_x', 3: '_y'}

    for sem_pair, form_pair, sem_diff, form_diff in zip(
        split_sem_pairs,
        split_form_pairs,
        index_sem_difference_sets,
        index_form_difference_sets
    ):
        # print(f"Processing pair: sem_pair={sem_pair}, form_pair={form_pair}") # ------------------------------------------------1114
       
        # 既存のラベルを探すロジック
        existing_label = None
        if form_diff:
            for form_elements in form_pair:
                for element in form_elements:
                    if '/' in element:
                        label, _ = element.split('/')
                        # 使用済みラベルを優先して再利用
                        if label in used_labels:
                            existing_label = label
                            break
                if existing_label is not None:
                    break

        # 既存ラベルが見つかった場合、それを使用し、見つからなければ新たに生成
        if existing_label is not None:
            label = existing_label
        else:
            label = generate_random_label(used_labels)
            used_labels.add(label)

        # print(f"Chosen label: {label}") # ------------------------------------------------1114

        # 適切な変数名を選択
        index = sem_diff[0]
        var = index_to_var.get(index, '_x')

        # スキーマルールの作成
        sem_of_scheme_rule = sem_pair[0][:]
        sem_of_scheme_rule[index] = var

        form_of_scheme_rule = form_pair[0][:]  # コピーして変更を加える
        if form_diff[0] < len(form_of_scheme_rule):
            form_of_scheme_rule[form_diff[0]] = f'{label}/{var[1]}'
        else:
            # print(f"Error: form_diff[0] is out of bounds for form_of_scheme_rule!") # ------------------------------------------------1114
            continue

        scheme_rule = [sem_of_scheme_rule, form_of_scheme_rule]

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

# 従来のchunk

def chunk_learning(sentence_rule_set_with_cv, word_rule_set):

    semantic_set = set_semantics(sentence_rule_set_with_cv)
    split_semantic_elements_set = split_semantics_process(semantic_set)
    pairs_with_differences = count_sem_difference_process(split_semantic_elements_set)
    sem_pairs_with_only_one_difference = detect_sem_pairs_with_only_one_difference(split_semantic_elements_set)
    transformed_form_pairs = transformed_set_form(sem_pairs_with_only_one_difference, sentence_rule_set_with_cv)
    split_form_pairs = split_form_process(transformed_form_pairs)
    compare_form_pair_results = compare_forms_by_index_process(split_form_pairs)

    check_for_uppercase_warning(compare_form_pair_results)
    filtered_indices_set = get_filtered_indices([result for result, _, _ in compare_form_pair_results])
    can_chunk_rule_set = final_transformed_set_form(sem_pairs_with_only_one_difference, sentence_rule_set_with_cv, filtered_indices_set)

    transform_only_sem_chunk_pair_sets = transform_only_sem_chunk_pair(can_chunk_rule_set)
    split_sem_pairs = split_sem_pairs_for_chunk(transform_only_sem_chunk_pair_sets)
    index_sem_difference_sets = detect_index_sem_difference_process(split_sem_pairs)
    transform_only_form_chunk_pair_sets = transform_only_form_chunk_pair(can_chunk_rule_set)
    split_form_pairs = split_form_process(transform_only_form_chunk_pair_sets)
    index_form_difference_sets = detect_index_form_difference_process(split_form_pairs)

    used_labels = used_labels_in_word_rule_set(word_rule_set)
    chunk_completed_pairs = chunk_completed_to_generate_scheme_rules_and_word_rules(
        split_sem_pairs,
        split_form_pairs,
        index_sem_difference_sets,
        index_form_difference_sets,
        word_rule_set
    )

    chunked_rules = []
    for scheme_rule, *word_rules in chunk_completed_pairs:
        sem_scheme_rule = scheme_rule[0]
        if len(sem_scheme_rule) >= 4:
            combined_element = f"({sem_scheme_rule[2]},{sem_scheme_rule[3]})"
            sem_scheme_rule = sem_scheme_rule[:2] + [combined_element] + sem_scheme_rule[4:]
        sem_scheme_rule = f"{sem_scheme_rule[0]}/" + "".join(sem_scheme_rule[1:])
        form_scheme_rule = "".join(scheme_rule[1])

        chunked_rules.append(f"{sem_scheme_rule}->{form_scheme_rule}")

        for word_rule in word_rules:
            sem_word_rule = "/".join(word_rule[0])
            form_word_rule = "".join(word_rule[1])
            chunked_rules.append(f"{sem_word_rule}->{form_word_rule}")

    chunk_applications = len(can_chunk_rule_set)
    remaining_rules = sentence_rule_set_with_cv[:]  # 修正箇所
    used_rules = [rule for rule_pair in can_chunk_rule_set for rule in rule_pair]
    unapplied_rules = [rule for rule in remaining_rules if rule not in used_rules]
    chunked_rules.extend(unapplied_rules)
    chunked_rules.extend(word_rule_set)

    return chunked_rules, chunk_applications

# CV付きchunk：1回目だけ推論

# def chunk_learning_with_cv_for_the_first_time(rule_set):
#     """
#     ルールセットを値ごとに分割し、各値に対して独立してチャンク学習を適用します。
#     """
    
#     # 1. ルールを分割
#     word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
    
#     # 2. 推論して文生成ルールをcv別に分割
#     cv_0_rule_set, cv_1_rule_set = separate_rule_set_by_cv(sentence_rule_set)
#     print('chunk_learning_with_cv_for_the_first_time - cv0ルールセット:', cv_0_rule_set)
#     print('chunk_learning_with_cv_for_the_first_time - cv1ルールセット:', cv_1_rule_set)

    
#     # 3. 各cv値に対して独立してチャンク学習を適用
#     chunked_rules_0, chunk_applications_0 = chunk_learning(cv_0_rule_set, word_rule_set)
#     chunked_rules_1, chunk_applications_1 = chunk_learning(cv_1_rule_set, word_rule_set)

#    # 4. 結果を統合して出力
#     combined_rules = chunked_rules_0 + chunked_rules_1
#     total_chunk_applications = chunk_applications_0 + chunk_applications_1

#     return combined_rules, total_chunk_applications

def chunk_learning_with_cv_for_the_first_time(rule_set):
    word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
    # print("初回チャンク: 文ルールセット:", sentence_rule_set) # --------------------------------1115
    # print("初回チャンク: 単語ルールセット:", word_rule_set) # --------------------------------1115
    cv_0_rule_set, cv_1_rule_set = separate_rule_set_by_cv(sentence_rule_set)
    # print("初回チャンク: cv0 ルールセット:", cv_0_rule_set) # --------------------------------1115
    # print("初回チャンク: cv1 ルールセット:", cv_1_rule_set) # --------------------------------1115

    chunked_rules_0, chunk_applications_0 = chunk_learning(cv_0_rule_set, word_rule_set)
    # print("初回チャンク: cv0 チャンク結果:", chunked_rules_0) # --------------------------------1115
    chunked_rules_1, chunk_applications_1 = chunk_learning(cv_1_rule_set, word_rule_set)
    # print("初回チャンク: cv1 チャンク結果:", chunked_rules_1) # --------------------------------1115

    combined_rules = chunked_rules_0 + chunked_rules_1
    total_chunk_applications = chunk_applications_0 + chunk_applications_1
    # print("初回チャンク: 結合後のルール:", combined_rules) # --------------------------------1115
    return combined_rules, total_chunk_applications


# CV付きchunk：2回目以降は推論しない

# def chunk_learning_with_cv_after_the_second_time(rule_set):
    
#     word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
#     cv_0_rule_set, cv_1_rule_set = simply_separate_rule_set_by_cv(sentence_rule_set)
#     print('2回目：cv0とcv1のリスト', cv_0_rule_set, cv_1_rule_set)
    
#     # 3. 各cv値に対して独立してチャンク学習を適用
#     chunked_rules_0, chunk_applications_0 = chunk_learning(cv_0_rule_set, word_rule_set)
#     chunked_rules_1, chunk_applications_1 = chunk_learning(cv_1_rule_set, word_rule_set)

#    # 4. 結果を統合して出力
#     combined_rules = chunked_rules_0 + chunked_rules_1
#     total_chunk_applications = chunk_applications_0 + chunk_applications_1

#     return combined_rules, total_chunk_applications

def chunk_learning_with_cv_after_the_second_time(rule_set):
    word_rule_set, sentence_rule_set = clustering_rule_set(rule_set)
    # print("2回目以降チャンク: 文ルールセット:", sentence_rule_set) # --------------------------------1115
    # print("2回目以降チャンク: 単語ルールセット:", word_rule_set) # --------------------------------1115
    cv_0_rule_set, cv_1_rule_set = simply_separate_rule_set_by_cv(sentence_rule_set)
    # print("2回目以降チャンク: cv0 ルールセット:", cv_0_rule_set) # --------------------------------1115
    # print("2回目以降チャンク: cv1 ルールセット:", cv_1_rule_set) # --------------------------------1115

    chunked_rules_0, chunk_applications_0 = chunk_learning(cv_0_rule_set, word_rule_set)
    # print("2回目以降チャンク: cv0 チャンク結果:", chunked_rules_0) # --------------------------------1115
    chunked_rules_1, chunk_applications_1 = chunk_learning(cv_1_rule_set, word_rule_set)
    # print("2回目以降チャンク: cv1 チャンク結果:", chunked_rules_1) # --------------------------------1115

    combined_rules = chunked_rules_0 + chunked_rules_1
    total_chunk_applications = chunk_applications_0 + chunk_applications_1
    # print("2回目以降チャンク: 結合後のルール:", combined_rules) # --------------------------------1115
    return combined_rules, total_chunk_applications