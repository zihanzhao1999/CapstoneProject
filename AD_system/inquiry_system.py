import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from DPTreeNode import DPTreeNode
import torch
from utils import read_zst, parse_cols, calculate_top_entropy_columns, parse_age
import os


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", help="intput path to the patient data file (zst file)", type=str)
    parser.add_argument(
        "--output_dir", help="output folder", type=str, default="data")
    args = parser.parse_args()
    
    # Load and process train data
    input_wo_sex = read_zst(args.input_path)
    # sex will not be consider when inquiring (based on EDA result)
    input_wo_sex = input_wo_sex.drop('SEX', axis=1).sort_index(axis=1)
    input_wo_sex['AGE'] = input_wo_sex['AGE'].apply(parse_age)

    n = len(input_wo_sex)
    # create input tensor
    input_wo_sex_tensor = torch.tensor(input_wo_sex.values, dtype=torch.int8).cuda()
    age_idx = input_wo_sex.columns.get_loc('AGE')
    # store the column names
    original_column_names = input_wo_sex.columns.copy()
    input_wo_sex.columns = range(len(input_wo_sex.columns))
    # create a tensor for input without age since age is not a question for inquiring,
    # but should be considered in the process 
    input_wo_age = input_wo_sex.copy()
    input_wo_age[age_idx] = 0

    input_wo_age_tensor = input_wo_sex_tensor.clone()
    input_wo_age_tensor[:, age_idx] = 0

    input_wo_sex = input_wo_sex.replace(0, -1) # change -1 to represent negative

    column_to_prefix, prefix_to_columns = parse_cols(input_wo_sex.columns)
    
    # Use a decision tree structure for dynamic programming
    # If there is already a inquiry system decision tree, then load it
    if os.path.isfile(f'{args.output_dir}/inquiry_system.pkl'):
        tree_root = pickle.load(open(f'{args.output_dir}/inquiry_system.pkl', 'rb'))
        print('Tree loaded!')
    else: # else create a tree root
        reveal, en = calculate_top_entropy_columns(input_wo_age_tensor, list(input_wo_sex.columns), prefix_to_columns, column_to_prefix)
        tree_root = DPTreeNode(None, reveal)
        tree_root.entropy = en

    result = None
    small_chunk = None
    large_chunk = None
    stored_tree_param = 10000 # store the tree per 10000 instances
    num_question = 20 # how many question to ask
    when_to_consider_age = 3 # consider age after 3+1 questions

    for idx in tqdm(range(len(input_wo_sex))): # for every data in the dataset
        
        instance_tensor = input_wo_sex_tensor[idx]
        instance_wo_age = input_wo_age_tensor[idx]
        
        # most 500/10000 similar data in the dataset are considered 
        # for asking the second question
        percent = 500 
        root = tree_root
        reveal = []
        similarity = None
        
        
        if idx == stored_tree_param:
            # Store the tree
            pickle.dump(root, open('inquiry_system.pkl', 'wb'))
            stored_tree_param += 10000
        
        for i in range(num_question): # for each question
            
            new_inquiry = root.next_question
            # Write the patient's answer
            if column_to_prefix[new_inquiry]:
                new_inquiry = prefix_to_columns[column_to_prefix[new_inquiry]]
            else:
                new_inquiry = [new_inquiry]
                
            reveal += new_inquiry
            # Write the patient's answer (in tree node)
            if i == when_to_consider_age:
                answer = "".join(str(x) for x in instance_tensor[new_inquiry + [age_idx]].tolist())
                reveal += [age_idx]
            else:
                answer = "".join(str(x) for x in instance_tensor[new_inquiry].tolist())
            
            # If it's already calculated, then move to the next question using tree    
            next_node = root.look_up_children(answer)
            if next_node:
                root = next_node
            else: # Calculate the next question to ask
                # If it's the last question, then don't calculate the next question
                if i == num_question - 1: continue
                # Create a new tree node
                cur_node = DPTreeNode(answer, None)
                root.add_child(cur_node)
                if similarity is not None: # Use similarity from last iteration
                    if i == when_to_consider_age:
                        for question_idx in new_inquiry:
                            similarity += 2 * (input_wo_age_tensor[:, question_idx] != instance_wo_age[question_idx])
                        similarity += (input_wo_sex_tensor[:, age_idx] - instance_tensor[age_idx]).abs()
                    else: 
                        for question_idx in new_inquiry:
                            similarity += 2 * (input_wo_age_tensor[:, question_idx] != instance_wo_age[question_idx])
                else: # Calculate similarity from the first question
                    similarity = torch.zeros(input_wo_sex_tensor.size(0), device="cuda")
                    for question_idx in reveal:
                        if question_idx == age_idx:
                            similarity += (input_wo_sex_tensor[:, age_idx] - instance_tensor[age_idx]).abs()
                        else:
                            similarity += 2 * (input_wo_age_tensor[:, question_idx] != instance_wo_age[question_idx])
                
                # Calculate how many most similar data to be consider 
                k = n//10000*percent
                values, top_indices = torch.topk(similarity, k, largest=False)
                top_actual_indices = top_indices
                similar_cases = input_wo_age_tensor[top_actual_indices]
                selected_col = [col for col in input_wo_sex.columns if col not in reveal]
                # Calculate the most largest entropy column(column group) within similar cases as the next question
                new_inquiry, en = calculate_top_entropy_columns(
                    similar_cases[:, selected_col], selected_col, prefix_to_columns, column_to_prefix)
                cur_node.next_question = new_inquiry
                cur_node.entropy = np.int8(en//2)
                root = cur_node

            # Calculate what (num/10000) most similar data to be consider for the next question
            percent = percent//(1.4**(root.entropy + 1))
            percent = int(percent) if percent > 1 else 1
            torch.cuda.synchronize()
        
        # Store the patient data (after inquriy)
        mask_instance = pd.Series(0, index=range(len(original_column_names))) # 0 represents unknown
        mask_instance[reveal] = input_wo_sex.iloc[idx][reveal]
        
        # Use chunks to save time
        if small_chunk is None:
            small_chunk = np.array([mask_instance.to_numpy()])
        else:
            small_chunk = np.vstack((small_chunk, mask_instance.to_numpy()))
        if idx % 100 == 99:
            if large_chunk is None:
                large_chunk = small_chunk
            else:
                large_chunk = np.vstack((large_chunk, small_chunk))
            small_chunk = None
            if idx % 10000 == 9999:
                if result is None:
                    result = large_chunk
                else:
                    result = np.vstack((result, large_chunk))
                large_chunk = None
        torch.cuda.synchronize()
                
    # Empty the chunks
    if large_chunk is not None:
        if result is None: 
            result = large_chunk
        else:
            result = np.vstack((result, large_chunk))
    if small_chunk is not None:
        if result is None: 
            result = small_chunk
        else:
            result = np.vstack((result, small_chunk))
            
    # Store data after inquiry and the decision tree
    result = pd.DataFrame(result, columns=original_column_names)
    result_file = f'{args.output_dir}/inquiry_result.zst'
    result.astype(np.int8).to_pickle(result_file)
    pickle.dump(tree_root, open(f'{args.output_dir}/inquiry_system.pkl', 'wb'))