import os
import random
from constants_video import *
import json
import argparse
import copy 


def get_sample_description(sample, properties, use_unstructured):
    if use_unstructured:
        description = DESCRIPTIONS[sample][0] + " Overall, it presents a"
    else:
        description = "It presents a"
    assert len(properties) >= 1
    if "hardness" in properties:
        description += f" {HARDNESS_MAP[RANKS['hardness'][sample]]}"
    if "hardness" in properties and "protrusion" in properties:
        description += f" and"
    if "protrusion" in properties:
        description += f" {PROTRUSION_MAP[RANKS['protrusion'][sample]]}"
    description += " surface"
    if "elasticity" in properties:
        description += f" with {ELASTICITY_MAP[RANKS['elasticity'][sample]]}"
    if "friction" in properties:
        description += f", and {FRICTION_MAP[RANKS['friction'][sample]]}"
    description += "."
    return description


def get_property_description_from_ranks(sample, properties):
    """Generates description based purely on ranks, avoiding object names in the core description."""
    # Ensure properties is a list
    if not isinstance(properties, list):
        properties = list(properties)

    parts = []
    # Surface properties
    surface_props = []
    if "hardness" in properties:
        surface_props.append(f"{HARDNESS_MAP[RANKS['hardness'][sample]]}")
    if "protrusion" in properties:
        surface_props.append(f"{PROTRUSION_MAP[RANKS['protrusion'][sample]]}")

    if surface_props:
        parts.append("a " + " and ".join(surface_props) + " surface")

    # Other properties
    other_props = []
    if "elasticity" in properties:
        other_props.append(f"{ELASTICITY_MAP[RANKS['elasticity'][sample]]}")
    if "friction" in properties:
        other_props.append(f"{FRICTION_MAP[RANKS['friction'][sample]]}")

    if other_props:
        connector = " with " if surface_props else ""
        parts.append(connector + " and ".join(other_props))

    if not parts:
        return "It has some generic tactile properties."

    description = "It presents " + "".join(parts) + "."
    description = description.replace("presents  with", "presents with") 
    description = description.replace("  ", " ")
    return description


def generate_one_step_qa(start_prompt, json_path, data_path, split, num_samples, use_unstructured, use_properties):
    properties = ["hardness", "protrusion", "elasticity", "friction"]

    property_names = {
        "hardness": "hardness",
        "protrusion": "protrusion",
        "elasticity": "elasticity",
        "friction": "friction"
    }

    # prompt setup
    property_comparisons = {
        "hardness": {
            "<more_property>": "harder",
            "<less_property>": "softer",
            "<most_property>": "hardest",
            "<least_property>": "softest"
        },
        "protrusion": {
            "<more_property>": "more protruded",
            "<less_property>": "less protruded",
            "<most_property>": "most protruded",
            "<least_property>": "least protruded"
        },
        "elasticity": {
            "<more_property>": "more elastic",
            "<less_property>": "less elastic",
            "<most_property>": "most elastic",
            "<least_property>": "least elastic"
        },
        "friction": {
            "<more_property>": "rougher",
            "<less_property>": "smoother",
            "<most_property>": "roughest",
            "<least_property>": "smoothest"
        }
    }

    # load all samples
    all_samples = {}
    for p in json_path:
        with open(p) as json_file:
            samples = json.load(json_file)
            json_file.close()
        for k, v in samples.items():
            if k in all_samples.keys():
                all_samples[k] += v
            else:
                all_samples[k] = v

    # prepare QA data
    all_data = []
    count = 0

    # 创建 5 种不同任务的 prompts
    tactile_feature_assessment = [{
        "tactile_feature_assessment_0": ["Describe the physical properties of the object in the video <video_start>", "<video_tokens>", "<video_end>."],
        "tactile_feature_assessment_1": ["How does the object in this tactile video <video_start>", "<video_tokens>", "<video_end> feel?"],
        "tactile_feature_assessment_2": ["Can you detail the surface characteristics shown in this video <video_start>", "<video_tokens>", "<video_end>?"],
        "tactile_feature_assessment_3": ["What are the tactile features of the object presented in the video <video_start>", "<video_tokens>", "<video_end>?"],
    }]

    surface_feature_distinction = [{
        "surface_feature_distinction_more_0": ["I have tactile videos of two objects. Which one is <more_property>? <video_start>", "<video_tokens>", "<video_end> <video_start>", "<video_tokens>", "<video_end>"],
        "surface_feature_distinction_less_0": ["I have tactile videos of two objects. Which one is <less_property>? <video_start>", "<video_tokens>", "<video_end> <video_start>", "<video_tokens>", "<video_end>"],
        "surface_feature_distinction_more_1": ["Between these two videos, <video_start>", "<video_tokens>", "<video_end> and <video_start>", "<video_tokens>", "<video_end>, which object feels <more_property>?"],
        "surface_feature_distinction_less_1": ["Comparing the objects in <video_start>", "<video_tokens>", "<video_end> and <video_start>", "<video_tokens>", "<video_end>, which one is <less_property>?"],
        "surface_feature_distinction_more_desc_0": ["Is the object in the first video <video_start>", "<video_tokens>", "<video_end> <more_property> than the one in the second video <video_start>", "<video_tokens>", "<video_end>? Describe both objects before answering."],
        "surface_feature_distinction_less_desc_0": ["Is the object in the first video <video_start>", "<video_tokens>", "<video_end> <less_property> than the one in the second video <video_start>", "<video_tokens>", "<video_end>? Describe both objects before answering."],
    }]

    surface_optimality_identification = [{
        "surface_optimality_identification_most_0": ["Given three tactile videos: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>. Describe each object and then select the <most_property> one."],
        "surface_optimality_identification_least_0": ["Given these tactile videos: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>. Describe each object and then select the <least_property> one."],
        "surface_optimality_identification_most_1": ["You have tactile videos of three objects: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>. Which object is the <most_property>? Please describe all three first."],
        "surface_optimality_identification_least_1": ["Among these three videos: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>, identify the <least_property> object after describing each one."],
    }]

    object_sensation_correlation = [{
        "object_sensation_correlation_0": ["Given three tactile videos: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>. Describe the object in each video, then match each video (a, b, c) to one of the following objects in alphabetical order: "],
        "object_sensation_correlation_1": ["You have tactile videos of three different objects: a) <video_start>", "<video_tokens>", "<video_end>, b) <video_start>", "<video_tokens>", "<video_end>, c) <video_start>", "<video_tokens>", "<video_end>. First, describe the properties shown in each video. Then, assign each video (a, b, c) to one of these objects listed alphabetically: "],
    }]


    if split == "train":
        property_questions = {
            f"{split}_surface_feature_distinction": surface_feature_distinction,
            f"{split}_surface_optimality_identification": surface_optimality_identification,
            f"{split}_object_sensation_correlation": object_sensation_correlation,
        }
        if use_properties: 
             property_questions[f"{split}_tactile_feature_assessment"] = tactile_feature_assessment
    elif split == "eval":
        property_questions = {
            f"{split}_tactile_feature_assessment": tactile_feature_assessment,
            f"{split}_surface_feature_distinction": surface_feature_distinction,
            f"{split}_surface_optimality_identification": surface_optimality_identification,
            f"{split}_object_sensation_correlation": object_sensation_correlation,
        }
    else: 
        property_questions = {
            f"{split}_surface_feature_distinction": surface_feature_distinction,
            f"{split}_surface_optimality_identification": surface_optimality_identification,
            f"{split}_object_sensation_correlation": object_sensation_correlation,
        }
        if use_properties:
             property_questions[f"{split}_tactile_feature_assessment"] = tactile_feature_assessment


 
    existing = {}
    if split == "eval":
        for key in property_questions.keys():
            existing[key] = set()

    max_retries = num_samples * 5 
    retries = 0

    # generate QA pairs
    while count < num_samples:
        if retries > max_retries:
            print(f"Warning: Could not generate {num_samples} unique samples after {max_retries} retries. Stopping generation for {split} split at {count} samples.")
            break

        question_type = random.choice(list(property_questions.keys()))
        available_prompts = property_questions[question_type][0]
        question_key = random.choice(list(available_prompts.keys()))
        question_template = available_prompts[question_key].copy() 
        question_steps = 1 

        key_to_add = None 
        video_paths_for_data = [] 
        answer = "" 
        question_suffix = "" 

        if question_type == f"{split}_tactile_feature_assessment":
            sample = random.choice(list(all_samples.keys()))
            if not all_samples[sample]: 
                 retries += 1
                 continue
            video_path = random.choice(all_samples[sample])

            if split == "eval":
                key_to_check = video_path
                if key_to_check in existing[question_type]:
                    retries += 1
                    continue 
                else:
                    key_to_add = key_to_check 

           
            answer = get_property_description_from_ranks(sample, properties)
            video_paths_for_data = [video_path]

        elif question_type == f"{split}_surface_feature_distinction":
            num_video = 2
            if len(all_samples) < num_video:
                retries += 1
                continue 
            selected_samples = random.sample(list(all_samples.keys()), k=num_video)
            prop = random.choice(properties)
            direction = "more" if "more" in question_key else "less"
           
            try:
                video_path_a = random.choice(all_samples[selected_samples[0]])
                video_path_b = random.choice(all_samples[selected_samples[1]])
            except IndexError:
                 retries += 1
                 continue 

            video_paths = [video_path_a, video_path_b]
            prop_description_term = property_comparisons[prop][f"<{direction}_property>"]

            if split == "eval":
                key_to_check = tuple(sorted(video_paths)) + (prop_description_term,)
                if key_to_check in existing[question_type]:
                    retries += 1
                    continue 
                else:
                    key_to_add = key_to_check

       
            rank_a = RANKS[prop][selected_samples[0]]
            rank_b = RANKS[prop][selected_samples[1]]
            core_answer = ""
          
            if rank_a == rank_b:
                # print(f"Skipping comparison due to identical ranks: {selected_samples[0]} ({rank_a}) vs {selected_samples[1]} ({rank_b}) for {prop}") # Optional debug
                retries += 1
                continue 
            
            if (direction == "more" and rank_a > rank_b) or (direction == "less" and rank_a < rank_b):
                 core_answer = "Yes."
            else: 
                 core_answer = "No."

            
            if "desc" in question_key and use_properties:
                desc1 = get_property_description_from_ranks(selected_samples[0], properties)
                desc2 = get_property_description_from_ranks(selected_samples[1], properties)
               
                answer = f"Object 1: {desc1} Object 2: {desc2} Conclusion: {core_answer}"
            else:
                if use_properties:
                    desc1 = get_property_description_from_ranks(selected_samples[0], properties)
                    desc2 = get_property_description_from_ranks(selected_samples[1], properties)
           
                    if (direction == "more" and rank_a > rank_b) or (direction == "less" and rank_a < rank_b):
                        simple_conclusion = "The first object."
                    else:
                        simple_conclusion = "The second object."
               
                    answer = f"Object 1: {desc1} Object 2: {desc2} Conclusion: {simple_conclusion}"
                else: 
                    if (direction == "more" and rank_a > rank_b) or (direction == "less" and rank_a < rank_b):
                        answer = "The first object."
                    else: 
                        answer = "The second object."

         
            if not answer or not isinstance(answer, str) or answer.strip() == "":
                 print(f"Warning: Generated empty or invalid answer for comparison between {selected_samples[0]} ({rank_a}) and {selected_samples[1]} ({rank_b}) on property {prop}. Original core_answer: '{core_answer}'. Skipping this sample.")
                 retries += 1 
                 continue
       

            video_paths_for_data = video_paths

        elif question_type == f"{split}_surface_optimality_identification":
            num_video = 3
            if len(all_samples) < num_video:
                retries += 1
                continue
            selected_samples = []
            attempts = 0
            while len(selected_samples) < num_video and attempts < 20:
                 candidate = random.choice(list(all_samples.keys()))
                 if candidate not in selected_samples:
                      selected_samples.append(candidate)
                 attempts += 1
            if len(selected_samples) < num_video:
                 retries += 1
                 continue

            prop = random.choice(properties)
            direction = "most" if "most" in question_key else "least"
            try:
                video_paths = [random.choice(all_samples[s]) for s in selected_samples]
            except IndexError:
                 retries += 1
                 continue

            prop_description_term = property_comparisons[prop][f"<{direction}_property>"]

            if split == "eval":
                key_to_check = tuple(sorted(video_paths)) + (prop_description_term,)
                if key_to_check in existing[question_type]:
                    retries += 1
                    continue
                else:
                    key_to_add = key_to_check

            ranks = {s: RANKS[prop][s] for s in selected_samples}

            rank_values = list(ranks.values())
            if len(set(rank_values)) == 1:
                retries += 1
                continue 

      
            target_rank_value = max(rank_values) if direction == "most" else min(rank_values)
            target_rank_count = rank_values.count(target_rank_value)

            if target_rank_count > 1:
                
                retries += 1
                continue 

            target_sample = max(ranks, key=ranks.get) if direction == "most" else min(ranks, key=ranks.get)

            target_label = ""
            for i, sample in enumerate(selected_samples):
                if sample == target_sample:
                    target_label = chr(ord('a') + i) + ")"
                    break

            answer = ""
            if use_properties:
                for i, sample in enumerate(selected_samples):
                    label = chr(ord('a') + i) + ")"
                    desc = get_property_description_from_ranks(sample, properties)
                    answer += f"{label} {desc} "
                answer += f"Conclusion: The {prop_description_term} object is {target_label}."
            else:
                 answer = f"The {prop_description_term} object is {target_label}."
            video_paths_for_data = video_paths


        elif question_type == f"{split}_object_sensation_correlation":
            num_video = 3
            if len(all_samples) < num_video:
                retries += 1
                continue
            selected_samples = []
            attempts = 0
            while len(selected_samples) < num_video and attempts < 20:
                 candidate = random.choice(list(all_samples.keys()))
                 if candidate not in selected_samples:
                      selected_samples.append(candidate)
                 attempts += 1
            if len(selected_samples) < num_video:
                 retries += 1
                 continue

            try:
                video_paths = [random.choice(all_samples[s]) for s in selected_samples]
            except IndexError:
                 retries += 1
                 continue

            if split == "eval":
                key_to_check = tuple(sorted(video_paths))
                if key_to_check in existing[question_type]:
                    retries += 1
                    continue
                else:
                    key_to_add = key_to_check

           
            property_vectors = []
            properties_to_check = ["hardness", "protrusion", "elasticity", "friction"]
            for sample in selected_samples:
                try:
                    vector = tuple(RANKS[prop][sample] for prop in properties_to_check)
                    property_vectors.append(vector)
                except KeyError as e:
                    print(f"Warning: Missing rank for object {e} in object_sensation_correlation. Skipping sample.")
                    retries += 1 
                    property_vectors = None 
                    break
            
            if property_vectors is None: 
                continue

        
            if len(property_vectors) != len(set(property_vectors)): 
                retries += 1
                continue 

            object_names_for_suffix = sorted([OBJECTS.get(s, s) for s in selected_samples])
            question_suffix = "1) " + ", 2) ".join(object_names_for_suffix[:-1]) + f", 3) {object_names_for_suffix[-1]}."

            answer = ""
            obj_index = {0: "a)", 1: "b)", 2: "c)"}
            if use_properties:
                for i, sample in enumerate(selected_samples):
                    desc = get_property_description_from_ranks(sample, properties)
                    answer += f"{obj_index[i]} {desc} "
                answer += "Conclusion: "
            match_parts = []
            for i, sample in enumerate(selected_samples):
                 match_parts.append(f"{obj_index[i]} is {OBJECTS.get(sample, sample)}")
            answer += ", ".join(match_parts[:-1]) + f" and {match_parts[-1]}."

            video_paths_for_data = video_paths

        else:
            retries += 1
            continue


        final_question = []
        video_token_count = 0
        prop_defined = 'prop' in locals() 

        for chunk in question_template:
            if chunk == "<video_tokens>":
                final_question.append("<video>")
                video_token_count += 1
            elif prop_defined and ("<more_property>" in chunk or "<less_property>" in chunk or \
                 "<most_property>" in chunk or "<least_property>" in chunk):
                replaced_chunk = chunk
                for tag in ["<more_property>", "<less_property>", "<most_property>", "<least_property>"]:
                     if tag in replaced_chunk:
                          replaced_chunk = replaced_chunk.replace(tag, property_comparisons[prop][tag])
                final_question.append(replaced_chunk)
            else:
                final_question.append(chunk)

        if question_type == f"{split}_object_sensation_correlation":
             final_question.append(question_suffix)

        final_question.insert(0, start_prompt)

        if len(video_paths_for_data) != video_token_count:
             print(f"Warning: Mismatch video tokens ({video_token_count}) and paths ({len(video_paths_for_data)}) for {question_type} {question_key}. Skipping.")
             retries += 1
             continue

        data = [{
            "question_type": question_type,
            "question_steps": question_steps
        }]
        data.append({
            "role": "USER",
            "content": final_question,
            "video": video_paths_for_data
        })
        data.append({
            "role": "ASSISTANT",
            "content": [answer],
            "video": []
        })

        if split == "eval":
            existing[question_type].add(key_to_add)

        all_data.append(data)
        count += 1
        retries = 0 

        if count % 100 == 0 or count == num_samples: 
            print(f"{count}/{num_samples} completed for {split} split")


    if split == "eval":
        file_name = f"test_qa"
    else:
        file_name = f"{split}_qa"
    if not use_properties:
        file_name += "_no_properties"
    if not use_unstructured:
        file_name += "_no_unstructured"

    output_dir = data_path
    data_file_path = os.path.join(output_dir, f"{file_name}.json")
    print(f"Saving QA data to: {data_file_path}")
    with open(data_file_path, "w") as data_file:
        json.dump(all_data, data_file, indent=4)


def generate_tfa_evaluation_qa(start_prompt, json_path, data_path, split, use_unstructured):
    properties = ["hardness", "protrusion", "elasticity", "friction"]

    # load samples
    with open(json_path) as json_file:
        samples = json.load(json_file)
        json_file.close()
    
    all_data = []
    
    for k, v in samples.items():
        for video_path in v:
            question_type = "eval_tactile_feature_assessment"
            question_steps = 1
            data = [{
                "question_type": question_type,
                "question_steps": question_steps
            }]
            
            for qs in range(question_steps):
                question = ["Describe the physical properties of <video_start>", "<video>", "<video_end>."]
                sample = k
                
                answer = get_property_description_from_ranks(sample, properties)
                
                if qs == 0:
                    question.insert(0, start_prompt)
                    
                data.append({
                    "role": "USER",
                    "content": question,
                    "video": [video_path]
                })
                data.append({
                    "role": "ASSISTANT",
                    "content": [answer],
                    "video": []
                })
                
            all_data.append(data)
    

    file_name = f"{split}_tfa_qa"
    if not use_unstructured:
        file_name += "_no_unstructured"
    

    output_dir = data_path
    data_file_path = os.path.join(output_dir, f"{file_name}.json")
    print(f"Saving tfa evaluation QA data to: {data_file_path}")
    with open(data_file_path, "w") as data_file:
        json.dump(all_data, data_file, indent=4)



def calculate_match_score(object_name, target_properties):
    score = 0
    for prop, target_ranks in target_properties.items():
        if prop not in RANKS:
            continue 
        actual_rank = RANKS[prop].get(object_name, None) 
        if actual_rank is None:
            continue 

        if target_ranks == [-1] or actual_rank in target_ranks:
            score += 1
    return score


def generate_tsa_evaluation_qa(start_prompt, json_path, data_path, num_samples, use_unstructured, use_tactile, split="test"):
    all_samples = {}
    for p in json_path:
        with open(p) as json_file:
            samples = json.load(json_file)
            json_file.close()
        for k, v in samples.items():
            valid_videos = [video for video in v if video and isinstance(video, str) and os.path.exists(video)]
            if not valid_videos:
                print(f"Warning: No valid/existing videos found for object {k} in {p}. Skipping this object.")
                continue
            if k not in all_samples:
                all_samples[k] = []
            all_samples[k].extend(valid_videos)

    objects = list(all_samples.keys())
    if len(objects) < 2:
        print(f"Error: Need at least two distinct object types with valid videos to generate tsa evaluation QA. Found: {objects}")
        return

    all_data = []
    count = 0
    max_retries_per_sample = 30 

    while count < num_samples:
        retries = 0
        while retries < max_retries_per_sample:
            if not SCENARIOS:
                print("Error: SCENARIOS list is empty in constants_video.py.")
                return
            scenario = random.choice(SCENARIOS)
            scenario_question = scenario['question']
            target_properties = scenario['target_properties']

            best_fit_object = None
            max_score = -1
            scores = {}
            available_objects = copy.deepcopy(objects) 
            random.shuffle(available_objects)

            for obj_name in available_objects:
                score = calculate_match_score(obj_name, target_properties)
                scores[obj_name] = score
                if score > max_score:
                    max_score = score
                    best_fit_object = obj_name
            
            if best_fit_object is None:
                print(f"Warning: Could not determine best fit object for scenario: {scenario_question}. Retrying.")
                retries += 1
                continue


            potential_second_objects = [o for o in available_objects if o != best_fit_object]
            if not potential_second_objects:
                print(f"Warning: Only one object type ({best_fit_object}) available or suitable. Cannot create comparison pair. Retrying.")
                retries += 1
                continue
            second_object = random.choice(potential_second_objects)

            try:
                video_a_path = random.choice(all_samples[best_fit_object])
                video_b_path = random.choice(all_samples[second_object])
                object_a = best_fit_object
                object_b = second_object
                correct_label = "a"

           
                if random.random() < 0.5:
                    video_a_path, video_b_path = video_b_path, video_a_path
                    object_a, object_b = object_b, object_a
                    correct_label = "b" 

            except (IndexError, KeyError) as e:
                print(f"Warning: Error selecting videos for {best_fit_object} or {second_object}: {e}. Retrying.")
                retries += 1
                continue
            except Exception as e:
                 print(f"Unexpected error selecting videos: {e}. Retrying.")
                 retries += 1
                 continue

          
            try:
                properties_to_describe = ["hardness", "protrusion", "elasticity", "friction"]
                desc_a = get_property_description_from_ranks(object_a, properties_to_describe)
                desc_b = get_property_description_from_ranks(object_b, properties_to_describe)
            except KeyError as e:
                 print(f"Warning: Missing rank information for object {e} needed for description. Retrying.")
                 retries += 1
                 continue
            except Exception as e:
                 print(f"Unexpected error generating descriptions: {e}. Retrying.")
                 retries += 1
                 continue


   
            question_type = "eval_tactile_scenario_analysis"
            question_steps = 2

            data = [{
                "question_type": question_type,
                "question_steps": question_steps
            }]

         
            data.append({
                "role": "USER",
                "content": [
                    start_prompt, 
                    f"Describe these two videos based on their tactile properties: a) <video_start><video><video_end>, b) <video_start><video><video_end>."
                ],
                "video": [video_a_path, video_b_path]
            })

           
            data.append({
                "role": "ASSISTANT",
                "content": [
                    f"a) {desc_a} b) {desc_b}" 
                ],
                "video": []
            })

         
            data.append({
                "role": "USER",
                "content": [
                    scenario_question, 
                    "Select only one most appropriate object (a or b) for this scenario based on the physical property descriptions provided above. Use the format 'The most suitable object is x)'." # 指示格式
                ],
                "video": []
            })

            data.append({
                "role": "ASSISTANT",
                "content": [
                    f"The most suitable object is {correct_label})." 
                ],
                "video": []
            })

            all_data.append(data)
            count += 1
            if count % 10 == 0 or count == num_samples:
                 print(f"Generated {count}/{num_samples} tsa samples (multi-turn format).")
            break 

        if retries == max_retries_per_sample:
             print(f"Warning: Max retries reached for generating tsa sample {count+1}. Moving on.")

        if count >= num_samples:
             break 

    # 保存数据
    file_name = f"{split}_tsa_qa"
    output_dir = data_path
    data_file_path = os.path.join(output_dir, f"{file_name}.json")
    print(f"Saving tsa evaluation QA data (multi-turn format) to: {data_file_path}")
    with open(data_file_path, "w") as data_file:
        json.dump(all_data, data_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='vtvllm', help='data root directory')
    args = parser.parse_args()

    use_unstructured = True
    use_tactile = True 
    use_properties = True 
    start_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"

    video_data_path = os.path.join(args.data_path, "vbts_video")
    train_json_path = os.path.join(video_data_path, "train_samples.json")
    val_json_path = os.path.join(video_data_path, "val_samples.json")
    test_json_path = os.path.join(video_data_path, "test_samples.json")

    os.makedirs(video_data_path, exist_ok=True)

    print("Generating QA with videos...")
    generate_one_step_qa(start_prompt, [train_json_path], video_data_path, "train",10000, use_unstructured, use_properties)
    # generate_tsa_evaluation_qa(start_prompt, [train_json_path], video_data_path, 1000, use_unstructured, use_tactile, "train")
    generate_tfa_evaluation_qa(start_prompt, val_json_path, video_data_path, "val", use_unstructured)
    # generate_tfa_evaluation_qa(start_prompt, test_json_path, video_data_path, "test", use_unstructured)
    generate_one_step_qa(start_prompt, [test_json_path], video_data_path, "test", 500, use_unstructured, use_properties)
    # generate_tsa_evaluation_qa(start_prompt, [test_json_path], video_data_path, 50, use_unstructured, use_tactile, "test") 

    print("Done!")
