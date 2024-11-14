import re
import itertools
from difflib import SequenceMatcher
import string
import random
from itertools import product

def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()
    form = parts[1].strip()
    return [semantic_structure, form]

def clustering_rule_set(rule_set):
    holistic_rule_set = []
    generalization_rule_set_1 = []
    generalization_rule_set_2 = []
    generalization_rule_set_3 = []
    word_rule_set = []

    for rule in rule_set:
        semantic_structure, _ = parse_rule(rule)

        if not semantic_structure.startswith("S/"):
            word_rule_set.append(rule)
        else:
            p_count = semantic_structure.count("_p")
            x_count = semantic_structure.count("_x")
            y_count = semantic_structure.count("_y")
            total_variables = p_count + x_count + y_count

            if total_variables == 0:
                holistic_rule_set.append(rule)
            elif total_variables == 1:
                generalization_rule_set_1.append(rule)
            elif total_variables == 2:
                generalization_rule_set_2.append(rule)
            elif total_variables == 3:
                generalization_rule_set_3.append(rule)
                
    
    return holistic_rule_set, generalization_rule_set_1, generalization_rule_set_2, generalization_rule_set_3, word_rule_set

def set_pair_sem_form(rule_set):
    pair_sem_form_set = []
    for a_rule in rule_set:
        a_pair_sem_form = parse_rule(a_rule)
        pair_sem_form_set.append(a_pair_sem_form)
    return pair_sem_form_set

def initialize_rule_sets(rule_set):
    holistic_rule_set, generalization_rule_set_1, generalization_rule_set_2, generalization_rule_set_3, word_rule_set = clustering_rule_set(rule_set)
    
    holistic_pair_sem_form_set = set_pair_sem_form(holistic_rule_set)
    slot_rules_set = [set_pair_sem_form(generalization_rule) for generalization_rule in [generalization_rule_set_1, generalization_rule_set_2, generalization_rule_set_3]]
    word_pair_sem_form_set = set_pair_sem_form(word_rule_set)
    
    return holistic_pair_sem_form_set, slot_rules_set, word_pair_sem_form_set

def detect_category_label_in_slot_rules(slot_rules_set):
    variables = ['p', 'x', 'y']
    detected_results = []  # 全ての結果を格納するリスト

    for slot_rules in slot_rules_set:
        for a_slot_rule in slot_rules:
            list1, list2 = a_slot_rule
            detected_varialbes_with_labels = []  # 変数全体： 例えば，　T/x
            detected_category_labels = [] # ラベルのみ 例えば， T

            for variable in variables:
                index = list2.find(variable)
                if index != -1 and index >= 2:  # 変数が見つかり、前に少なくとも2文字ある場合
                    detected_variables_with_category_label = list2[index-2:index+1] # 変数全体： 例えば，　T/x
                    detected_varialbes_with_labels.append(detected_variables_with_category_label)
                    detected_category_label = list2[index-2] # ラベルのみ 例えば， T
                    detected_category_labels.append(detected_category_label)


            # 各ルールの結果をリストに追加
            if detected_variables_with_category_label:  # 検出されたラベルがある場合のみ追加
                detected_results.append([a_slot_rule, [detected_varialbes_with_labels, detected_category_labels]])

    return detected_results

def detect_word_rule_with_the_same_category_label(detected_results, word_pair_sem_form_set):
    all_detected_results = []  # 複数の detected_result の結果を格納するリスト

    for single_detected_result in detected_results:  # 各 detected_result に対して処理
        category_labels_after_index_2_set = set()
        
        # インデックス1以降のカテゴリラベルを抽出
        for category_labels in single_detected_result[1]:  
            for a_category_label in category_labels:
                category_labels_after_index_2_set.add(a_category_label)

        # 抽出したカテゴリラベルに基づき word_pair_sem_form_set をフィルタリング
        detected_word_rule_with_the_same_category_label_set = []
        for a_word_pair_sem_form_set in word_pair_sem_form_set:
            for a_category_labels_after_index_2 in category_labels_after_index_2_set:
                if a_category_labels_after_index_2 in a_word_pair_sem_form_set[0]:
                    detected_word_rule_with_the_same_category_label_set.append(a_word_pair_sem_form_set)
                    break  # 条件を満たしたら次のペアへ進む

        # 単一の detected_result の結果をリストに追加
        all_detected_results.append([single_detected_result, detected_word_rule_with_the_same_category_label_set])

    return all_detected_results

def combinate_sentence_rules_with_duplication(all_detected_results):
    complete_sentence_rules_with_duplication = []

    for detected_result in all_detected_results:
        # スロット付き文ルールとそのカテゴリーラベル
        sentence_rule, category_label_slots = detected_result[0]
        slot_candidates = detected_result[1]

        sentence_structure, form_structure = sentence_rule  # 文構造と形式構造
        slots_with_labels = category_label_slots[0]  # スロットの完全な表記（例：T/x）
        labels = category_label_slots[1]  # ラベルだけのリスト（例：T）

        # スロットのラベルに対応する候補をマッピング
        label_to_candidates = {}
        for candidate in slot_candidates:
            label = candidate[0][0]  # ラベル（例：T, D, W）
            if label not in label_to_candidates:
                label_to_candidates[label] = []
            label_to_candidates[label].append(candidate)

        # 各スロットに対応する候補の組み合わせを取得
        slot_filled_combinations = []
        for label in labels:
            if label in label_to_candidates:
                slot_filled_combinations.append(label_to_candidates[label])

        # 各スロット候補の全組み合わせを生成
        all_combinations = product(*slot_filled_combinations)

        # 各組み合わせでスロットを埋めて完全な文ルールを作成
        for combination in all_combinations:
            filled_sentence = sentence_structure
            filled_form = form_structure

            # スロットを埋める
            for slot, candidate in zip(slots_with_labels, combination):
                variable = f"_{slot[2]}"  # 変項（例：_x）
                candidate_text = candidate[0][2:]  # 候補のテキスト（例：alice）
                filled_sentence = filled_sentence.replace(variable, candidate_text)
                filled_form = filled_form.replace(slot, candidate[1])

            complete_sentence_rules_with_duplication.append([filled_sentence, filled_form])

    return complete_sentence_rules_with_duplication

def delete_duplication_in_complete_sentence_rules(complete_sentence_rules_with_duplication):
    unique_sentence_rules = {}
    
    for rule in complete_sentence_rules_with_duplication:
        meaning_expression = rule[0]  # 意味表現部分
        if meaning_expression not in unique_sentence_rules:
            # 重複がない場合はそのまま追加
            unique_sentence_rules[meaning_expression] = rule
        else:
            # 重複がある場合はランダムに保持または置き換え
            unique_sentence_rules[meaning_expression] = random.choice([unique_sentence_rules[meaning_expression], rule])

    # 結果をリスト形式で返す
    return list(unique_sentence_rules.values())

def intersection_set_with_semantic_space(unique_sentence_rules, semantic_space):
    semantic_space_set = set(semantic_space) # semantic_space をセットに変換して高速な検索ができるようにする

    intersection_rules = [
        rule for rule in unique_sentence_rules if rule[0] in semantic_space_set
    ]
    
    
    # デバッグ出力
#     print(f"Number of intersection rules: {len(intersection_rules)}")
#     print(f"Size of semantic space: {len(semantic_space)}")
    
    expressivity = len(intersection_rules) / len(semantic_space)
    
    return intersection_rules, expressivity

def expressivity(rule_set, semantic_space):
    # 1. ルールセットをクラスターに分ける
    holistic_rule_set, generalization_rule_set_1, generalization_rule_set_2, generalization_rule_set_3, word_rule_set = clustering_rule_set(rule_set)

    # 2. 初期化してペアを取得
    holistic_pair_sem_form_set, slot_rules_set, word_pair_sem_form_set = initialize_rule_sets(rule_set)

    # 3. スロットに基づいてカテゴリーラベルを検出
    detected_results = detect_category_label_in_slot_rules(slot_rules_set)
    
    # 4. カテゴリーラベルに基づいて単語ルールを検出
    all_detected_results = detect_word_rule_with_the_same_category_label(detected_results, word_pair_sem_form_set)

    # 5. スロットを埋めた重複付き文ルールを作成
    complete_sentence_rules_with_duplication = combinate_sentence_rules_with_duplication(all_detected_results)

    # 6. 重複を削除してユニークな文ルールを作成
    unique_sentence_rules = delete_duplication_in_complete_sentence_rules(complete_sentence_rules_with_duplication)

    # 7. 意味空間と一致するルールの積集合と expressivity を計算
    _, expressivity = intersection_set_with_semantic_space(unique_sentence_rules, semantic_space)

    # expressivity のみを返す
    return expressivity