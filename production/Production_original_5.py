import re
import itertools
from difflib import SequenceMatcher
import string
import random

# chunkから借用
def parse_rule(rule):
    parts = rule.split('->')
    semantic_structure = parts[0].strip()
    form = parts[1].strip()
    return semantic_structure, form

def set_semantics(rule_set):
    semantic_set = []
    for a_rule in rule_set:
        a_semantics = parse_rule(a_rule)[0]
        semantic_set.append(a_semantics)
    return semantic_set

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

def set_pair_sem_form_for_word_rule(rule_set):
    pair_sem_form_set = []
    for a_rule in rule_set:
        pair_sem_form_set.append(a_rule)
    return pair_sem_form_set

def initialize_rule_sets(rule_set):
    holistic_rule_set, generalization_rule_set_1, generalization_rule_set_2, generalization_rule_set_3, word_rule_set = clustering_rule_set(rule_set)

    holistic_rule_set = set_pair_sem_form(holistic_rule_set)
    variable_1_pair_sem_form_set = set_pair_sem_form(generalization_rule_set_1)
    variable_2_pair_sem_form_set = set_pair_sem_form(generalization_rule_set_2)
    variable_3_pair_sem_form_set = set_pair_sem_form(generalization_rule_set_3)
    word_rule_set = set_pair_sem_form_for_word_rule(word_rule_set)

    return holistic_rule_set, variable_1_pair_sem_form_set, variable_2_pair_sem_form_set, variable_3_pair_sem_form_set, word_rule_set

def split_semantics_ability(semantic_elements):
    return re.findall(r'_[a-zA-Z0-9]+|\(\w+\)|[A-Z]+|/[0-9]', semantic_elements)

def split_semantics_process_for_rule_set(semantic_set):
    split_semantic_elements_set_in_rule_set = []
    for a_semantic_element in semantic_set:
        a_sem_express = a_semantic_element[0]
        a_form_express = a_semantic_element[1]
        split_semantics = split_semantics_ability(a_sem_express)
        split_semantic_elements_set_in_rule_set.append([split_semantics, [a_form_express]])
    return split_semantic_elements_set_in_rule_set

def split_semantics_process(semantic_set):
    split_semantic_elements_set = []
    for a_semantic_element in semantic_set:
        one_of_semantic_set = split_semantics_ability(a_semantic_element)
        split_semantic_elements_set.append(one_of_semantic_set)
    return split_semantic_elements_set

def initialize_semantic_elements(rule_set):
    holistic_rule_set, variable_1_pair_sem_form_set, variable_2_pair_sem_form_set, variable_3_pair_sem_form_set, word_rule_set = initialize_rule_sets(rule_set)

    split_semantic_elements_set_in_holistic_rule_set = split_semantics_process_for_rule_set(holistic_rule_set)
    split_semantic_elements_set_in_generalization_rule_set_1 = split_semantics_process_for_rule_set(variable_1_pair_sem_form_set)
    split_semantic_elements_set_in_generalization_rule_set_2 = split_semantics_process_for_rule_set(variable_2_pair_sem_form_set)
    split_semantic_elements_set_in_generalization_rule_set_3 = split_semantics_process_for_rule_set(variable_3_pair_sem_form_set)

    return split_semantic_elements_set_in_holistic_rule_set, split_semantic_elements_set_in_generalization_rule_set_1, split_semantic_elements_set_in_generalization_rule_set_2, split_semantic_elements_set_in_generalization_rule_set_3

def count_sem_difference_ability(split_sem1, split_sem2):
    sem_differences = 0
    for sem_element1, sem_element2 in zip(split_sem1, split_sem2):
        if sem_element1 != sem_element2:
            sem_differences += 1
    return sem_differences

def compare_production_sem_with_any_rule_set_ability(a_split_semantic_elements_set_in_production, any_rule_set, allowed_variables):
    variables = []
    if not any_rule_set:
        return variables
    
    for an_element_in_any_rule_set in any_rule_set:
        number_of_variables = count_sem_difference_ability(a_split_semantic_elements_set_in_production, an_element_in_any_rule_set[0])
        if number_of_variables == allowed_variables:
            variables.append((an_element_in_any_rule_set, a_split_semantic_elements_set_in_production, number_of_variables))
    return variables

def pairing_production_and_rules_with_any_variables_process(
    split_semantic_elements_set_in_production,
    split_semantic_elements_set_in_holistic_rule_set,
    split_semantic_elements_set_in_generalization_rule_set_1,
    split_semantic_elements_set_in_generalization_rule_set_2,
    split_semantic_elements_set_in_generalization_rule_set_3
):
    any_diff_in_sem_pairs = []
    
    for a_split_semantic_elements_set_in_production in split_semantic_elements_set_in_production:
        
        holistic_pairs = compare_production_sem_with_any_rule_set_ability(a_split_semantic_elements_set_in_production, split_semantic_elements_set_in_holistic_rule_set, 0)
        if holistic_pairs:
            any_diff_in_sem_pairs.append(holistic_pairs)
            continue
        
        variable1_pairs = compare_production_sem_with_any_rule_set_ability(a_split_semantic_elements_set_in_production, split_semantic_elements_set_in_generalization_rule_set_1, 1)
        if variable1_pairs:
            any_diff_in_sem_pairs.append(variable1_pairs)
            continue
        
        variable2_pairs = compare_production_sem_with_any_rule_set_ability(a_split_semantic_elements_set_in_production, split_semantic_elements_set_in_generalization_rule_set_2, 2)
        if variable2_pairs:
            any_diff_in_sem_pairs.append(variable2_pairs)
            continue
        
        variable3_pairs = compare_production_sem_with_any_rule_set_ability(a_split_semantic_elements_set_in_production, split_semantic_elements_set_in_generalization_rule_set_3, 3)
        if variable3_pairs:
            any_diff_in_sem_pairs.append(variable3_pairs)
            continue
        
        # print(a_split_semantic_elements_set_in_production)
        any_diff_in_sem_pairs.append(a_split_semantic_elements_set_in_production)
    
    return any_diff_in_sem_pairs

def cluster_any_diff_in_sem_pairs(any_diff_in_sem_pairs):
    # 発話指令のみを格納するリスト
    only_command_list = []
    # 発話指令以外（3つの要素がある）のリスト
    command_with_info_list = []
    
    for item in any_diff_in_sem_pairs:
        # 発話指令だけの場合
        if isinstance(item, list) and all(isinstance(elem, str) for elem in item):
            only_command_list.append(item)
        # 既存知識、発話指令、difference_countが含まれる場合
        else:
            command_with_info_list.append(item)
    
    return only_command_list, command_with_info_list

def form_random_generation_ability(length):
    allowed_characters = ''.join(c for c in string.ascii_lowercase if c not in 'spxy')
    return ''.join(random.choice(allowed_characters) for _ in range(length))

def form_one_two_three_generation_ability(non_variable_number):
    random_form = form_random_generation_ability(random.randint(1, 3))
    return random_form

def generate_invented_rules(utterance_list):
    invented_rules_from_only_command_list = []

    for utterance in utterance_list:
        target_sem_express = '/'.join(utterance[:2])

        if len(utterance) > 3:
            target_sem_express += '(' + ','.join(utterance[2:-1]) + ')'  # 引数部分を追加
        target_sem_express += utterance[-1]  # 最後の部分 (/0)を追加
        
        # 文全体をランダムに生成 (3〜9文字)
        full_invention = form_random_generation_ability(random.randint(3, 9))
        
        # 発話指令に対して文全体の形式をランダムに適用
        invented_rule = f"{target_sem_express}->{full_invention}"
        
        # 生成したルールをリストに保存
        invented_rules_from_only_command_list.append(invented_rule)
    
    return invented_rules_from_only_command_list

def extract_different_elements_and_format_ability(any_diff_in_sem_pair):
    list1, list2, _ = any_diff_in_sem_pair

    list1_flat = [item for sublist in list1 for item in sublist]
    list2_flat = list2

    differences = []
    formatted_results = []
    variable_and_sem_express_pairs = []

    for i in range(len(list2_flat)):
        if list1_flat[i] != list2_flat[i]:
            if not list1_flat[i].startswith('_'):
                differences.append(list1_flat[i])
            differences.append(list2_flat[i])

            if list1_flat[i].startswith('_'):
                variable = list1_flat[i][1]
                for sublist in list1:
                    for item in sublist:
                        if variable in item:
                            index = item.index(variable)
                            if index >= 2:
                                formatted_item = item[index-2:index] + variable
                                formatted_results.append(formatted_item)
                                
                                formatted_combined = item[index-2:index] + list2_flat[i]
                                variable_and_sem_express_pairs.append([variable, formatted_combined])

    final_result = formatted_results + differences
    return final_result, variable_and_sem_express_pairs

def detect_and_invention_word_rule_ability(variable_and_sem_express_pairs, word_rule_set):
    word_variable_and_sem_express_pairs = []
    invented_rules = []
    generated_forms = {}
    
    for pair in variable_and_sem_express_pairs:
        matching_rules = []

        target_sem_express = pair[1]
        
        if target_sem_express in generated_forms:
            selected_rule = generated_forms[target_sem_express]
        else:
            for rule in word_rule_set:
                sem_express_in_rule = rule.split('->')[0]
                if sem_express_in_rule == target_sem_express:
                    matching_rules.append(rule)

            if len(matching_rules) == 1:
                selected_rule = matching_rules[0]
            elif len(matching_rules) > 1:
                selected_rule = random.choice(matching_rules)
            else:
                random_word_form = form_one_two_three_generation_ability(random.randint(1, 3))
                selected_rule = f"{target_sem_express}->{random_word_form}"
                invented_rules.append(selected_rule)
                generated_forms[target_sem_express] = selected_rule

        word_variable_and_sem_express_pairs.append([pair[0], selected_rule])
    
    return word_variable_and_sem_express_pairs, invented_rules

def process_single_diff_in_sem_pair_ability(any_diff_in_sem_pair, word_rule_set):
    list1, list2, difference_count = any_diff_in_sem_pair

    if difference_count == 0:
        return list1, [], 0, [], any_diff_in_sem_pair

    final_result, variable_and_sem_express_pairs = extract_different_elements_and_format_ability(any_diff_in_sem_pair)
    word_variable_and_sem_express_pairs, invented_rules = detect_and_invention_word_rule_ability(variable_and_sem_express_pairs, word_rule_set)

    if len(invented_rules) >= 2:
        full_invention = form_random_generation_ability(random.randint(3, 9))
        final_result = [full_invention]
        word_variable_and_sem_express_pairs = [[var, full_invention] for var, _ in word_variable_and_sem_express_pairs]
        invented_rules = []
        invention_count = 3
    else:
        invention_count = len(invented_rules)
    
    return final_result, word_variable_and_sem_express_pairs, invention_count, invented_rules, any_diff_in_sem_pair

def select_best_utterance(possible_pairs, word_rule_set):
    best_pair = None
    lowest_invented_rule_count = float('inf')
    best_invented_rules = []
    best_word_variable_and_sem_express_pairs = []

    for i, pair in enumerate(possible_pairs):
        final_result, word_variable_and_sem_express_pairs, invention_count, invented_rules, _ = process_single_diff_in_sem_pair_ability(pair, word_rule_set)

        if invention_count < lowest_invented_rule_count:
            lowest_invented_rule_count = invention_count
            best_pair = pair
            best_invented_rules = invented_rules
            best_word_variable_and_sem_express_pairs = word_variable_and_sem_express_pairs
        elif invention_count == lowest_invented_rule_count:
            if random.choice([True, False]):
                best_pair = pair
                best_invented_rules = invented_rules
                best_word_variable_and_sem_express_pairs = word_variable_and_sem_express_pairs

    return best_pair, best_invented_rules, best_word_variable_and_sem_express_pairs

def process_and_return_production_ingredients(any_diff_in_sem_pairs, word_rule_set):
    production_ingredients = []

    for any_diff_in_sem_pair_group in any_diff_in_sem_pairs:
        if isinstance(any_diff_in_sem_pair_group, list) and len(any_diff_in_sem_pair_group) > 1:
            best_pair, best_invented_rules, best_word_variable_and_sem_express_pairs = select_best_utterance(any_diff_in_sem_pair_group, word_rule_set)
            final_result, _, _, _, _ = process_single_diff_in_sem_pair_ability(best_pair, word_rule_set)
            
            if best_invented_rules:
                word_rule_set.extend(best_invented_rules)
            
            production_ingredient = [best_pair, best_word_variable_and_sem_express_pairs]
            production_ingredients.append(production_ingredient)
        else:
            final_result, best_word_variable_and_sem_express_pairs, _, best_invented_rules, best_pair = process_single_diff_in_sem_pair_ability(any_diff_in_sem_pair_group[0], word_rule_set)
            
            if best_invented_rules:
                word_rule_set.extend(best_invented_rules)
            
            production_ingredient = [best_pair, best_word_variable_and_sem_express_pairs]
            production_ingredients.append(production_ingredient)

    return production_ingredients

def combine_word_and_variable_ability(an_only_one_diff_in_sem_pair, word_variable_and_sem_express_pairs):
    meaning_part = an_only_one_diff_in_sem_pair[1]
    form_part = an_only_one_diff_in_sem_pair[0][1]
    new_meaning = meaning_part[:]
    new_form = ''.join(form_part)
    
    if len(word_variable_and_sem_express_pairs) == 2 and all(pair[1] == word_variable_and_sem_express_pairs[0][1] for pair in word_variable_and_sem_express_pairs):
        word_form = word_variable_and_sem_express_pairs[0][1]
        return restore_sentence_rule_ability([new_meaning, [word_form]])

    for i, element in enumerate(new_meaning):
        if element.startswith('_'):
            position = element[1]

            for var_pos, word_rule in word_variable_and_sem_express_pairs:
                if var_pos == position:
                    word_meaning, word_form = word_rule.split('->')
                    new_meaning[i] = word_meaning.split('/')[1]

    # for var_pos, word_rule in word_variable_and_sem_express_pairs:
    
    #     category_label_and_sem_express, word_meaning_form = word_rule.split('->')
    #     word_form = word_meaning_form
    #     category_label = category_label_and_sem_express.split('/')[0]
        
    #     new_form = new_form.replace(f'{category_label}/{var_pos}', word_form)
    

    for var_pos, word_rule in word_variable_and_sem_express_pairs:
    # ルールに '->' が含まれているか確認する
        if '->' not in word_rule:
            # print(f"Invalid word rule format: {word_rule}")  # デバッグ用出力
            continue  # 無視して次に進む
        
        category_label_and_sem_express, word_meaning_form = word_rule.split('->')
        word_form = word_meaning_form
        category_label = category_label_and_sem_express.split('/')[0]
    
        new_form = new_form.replace(f'{category_label}/{var_pos}', word_form)

    return restore_sentence_rule_ability([new_meaning, [new_form]])


def restore_sentence_rule_ability(split_sentence_rule):
    split_sem_express = split_sentence_rule[0]
    
    restored_sem_express = split_sem_express[0] + '/' + split_sem_express[1]

    if len(split_sem_express) > 3:
        restored_sem_express += '(' + ','.join(split_sem_express[2:-1]) + ')'
    
    restored_sem_express += split_sem_express[-1]
    
    form_express = split_sentence_rule[1][0]
    
    return f"{restored_sem_express}->{form_express}"



def generate_sentence_rule(best_pair, word_variable_and_sem_express_pairs):
    if not word_variable_and_sem_express_pairs:
        meaning_part = best_pair[1]
        form_part = best_pair[0][1]
        return f"{meaning_part[0]}/" + f"{meaning_part[1]}" + "(" + ",".join(meaning_part[2:-1]) + ")" + meaning_part[-1] + "->" + "".join(form_part)
    else:
        return combine_word_and_variable_ability(best_pair, word_variable_and_sem_express_pairs)

def generate_all_sentence_rules(production_ingredients):
    generated_rules = []
    
    for pair_and_vars in production_ingredients:
        best_pair, word_variable_and_sem_express_pairs = pair_and_vars
        rule = generate_sentence_rule(best_pair, word_variable_and_sem_express_pairs)
        generated_rules.append(rule)
    
    return generated_rules

def produce(rule_set, only_sem_express_set_for_production):
    # 初期化
    holistic_rule_set, variable_1_pair_sem_form_set, variable_2_pair_sem_form_set, variable_3_pair_sem_form_set, word_rule_set = initialize_rule_sets(rule_set)

    split_semantic_elements_set_in_holistic_rule_set = split_semantics_process_for_rule_set(holistic_rule_set)
    split_semantic_elements_set_in_generalization_rule_set_1 = split_semantics_process_for_rule_set(variable_1_pair_sem_form_set)
    split_semantic_elements_set_in_generalization_rule_set_2 = split_semantics_process_for_rule_set(variable_2_pair_sem_form_set)
    split_semantic_elements_set_in_generalization_rule_set_3 = split_semantics_process_for_rule_set(variable_3_pair_sem_form_set)
    
    split_semantic_elements_set_in_production = split_semantics_process(only_sem_express_set_for_production)

    # 発話指令とルールの対応付け
    any_diff_in_sem_pairs = pairing_production_and_rules_with_any_variables_process(
        split_semantic_elements_set_in_production,
        split_semantic_elements_set_in_holistic_rule_set,
        split_semantic_elements_set_in_generalization_rule_set_1,
        split_semantic_elements_set_in_generalization_rule_set_2,
        split_semantic_elements_set_in_generalization_rule_set_3
    )

    only_command_list, command_with_info_list = cluster_any_diff_in_sem_pairs(any_diff_in_sem_pairs)
    
    
    # 発話成分の生成
    invented_rules_from_only_command_list = generate_invented_rules(only_command_list)
    production_ingredients = process_and_return_production_ingredients(command_with_info_list, word_rule_set)
    

    # 生成された文規則
    generated_rules = generate_all_sentence_rules(production_ingredients)
    generated_rules.extend(invented_rules_from_only_command_list)

    return generated_rules


# ---------------------   実行　　　　　-------------------------------

# holistic_rule_set, variable_1_pair_sem_form_set, variable_2_pair_sem_form_set, variable_3_pair_sem_form_set, word_rule_set = initialize_rule_sets(rule_set)
# split_semantic_elements_set_in_production = split_semantics_process(only_sem_express_set_for_production)
# split_semantic_elements_set_in_holistic_rule_set, split_semantic_elements_set_in_generalization_rule_set_1, split_semantic_elements_set_in_generalization_rule_set_2, split_semantic_elements_set_in_generalization_rule_set_3 = initialize_semantic_elements(rule_set)
# generated_rules = produce(rule_set, only_sem_express_set_for_production)
