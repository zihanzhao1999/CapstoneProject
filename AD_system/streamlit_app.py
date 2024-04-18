import streamlit as st
import pickle
import pandas as pd
import numpy as np
from DPTreeNode import DPTreeNode
import torch
from utils import (read_zst, parse_cols, calculate_top_entropy_columns, parse_age,
                   column_translate, pathology_translate)
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
from LSTMModel import LSTMModel


if __name__ == '__main__':
    
    st.set_page_config(
        page_title="Auto Diagnosis",
        page_icon=".streamlit/favicon.png",
    )
    
    if 'question_num' not in st.session_state:
        st.session_state.question_num = 0
        # Load every needed in AD system
        # -------------------------------------------
        st.session_state.tree_root = pickle.load(open('./stored_files/simple_tree.pkl', 'rb'))
        input_size = 518  # number of features
        hidden_size = 64
        output_size = 49  # number of unique diagnoses
        st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.session_state.model = LSTMModel(input_size, hidden_size, output_size).to(st.session_state.device)
        st.session_state.model.load_state_dict(torch.load('./stored_files//best_model.pth'))
        st.session_state.model.eval()
        st.session_state.age_scaler = pickle.load(open('./stored_files/age_scaler.pkl', 'rb'))
        st.session_state.column_translations = column_translate()
        st.session_state.pathology_translations = pathology_translate()
        
        # Load and process train data (for the inquiry system to infer)
        st.session_state.train_wo_sex = read_zst('./stored_files/train_x.zst')
        st.session_state.train_wo_sex = st.session_state.train_wo_sex.drop('SEX', axis=1).sort_index(axis=1)
        st.session_state.train_age = st.session_state.train_wo_sex['AGE'].copy()
        st.session_state.train_wo_sex['AGE'] = st.session_state.train_wo_sex['AGE'].apply(parse_age)

        st.session_state.n = len(st.session_state.train_wo_sex)
        st.session_state.train_wo_sex_tensor = torch.tensor(st.session_state.train_wo_sex.values, dtype=torch.int8).to(st.session_state.device)
        st.session_state.age_idx = st.session_state.train_wo_sex.columns.get_loc('AGE')
        st.session_state.original_column_names = st.session_state.train_wo_sex.columns.copy()
        st.session_state.train_wo_sex.columns = range(len(st.session_state.train_wo_sex.columns))
        st.session_state.train_wo_age = st.session_state.train_wo_sex.copy()
        st.session_state.train_wo_age[st.session_state.age_idx] = 0

        st.session_state.train_wo_age_tensor = st.session_state.train_wo_sex_tensor.clone()
        st.session_state.train_wo_age_tensor[:, st.session_state.age_idx] = 0

        st.session_state.train_wo_sex = [] # clean up space
        st.session_state.train_wo_age = [] # clean up space
    
        # Parse column names
        st.session_state.prefix_to_columns, st.session_state.column_to_prefix = parse_cols(st.session_state.original_column_names)
        
        st.session_state.all_pathologies = pickle.load(open('./stored_files/all_pathologies.pkl', 'rb'))
        # -------------------------------------------
    
    # Title of the app
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image('.streamlit/AD_icon.png', width=100)

    with col2:
        st.title('Auto Diagnosis App')
    
    if st.session_state.question_num == 0:
        # User inputs for age and sex
        st.session_state.instance_tensor = torch.zeros(517, dtype=torch.int8).to(st.session_state.device)
        st.session_state.age = st.number_input('Enter your age', min_value=0, max_value=130, step=1)
        st.session_state.sex = st.selectbox('Select your sex', [''] + ['Male', 'Female', 'Other'])
        st.session_state.questions = []
        st.session_state.answers = []
        
        # Agent inquiries
        st.session_state.num_question = 20
        st.session_state.percent = 500
        st.session_state.when_to_consider_age = 3
        st.session_state.root = st.session_state.tree_root
        st.session_state.reveal = []
        st.session_state.similarity = None
        
    # When question number < number of questions in setting, keep asking
    if st.session_state.question_num < st.session_state.num_question:
        st.write(f'Question #{st.session_state.question_num +1}')
        new_inquiry = st.session_state.root.next_question
        col_english = st.session_state.column_translations[st.session_state.original_column_names[new_inquiry]]
        # Show the question
        st.write(col_english.split('_@_')[0])
        # Show the options
        if st.session_state.column_to_prefix[new_inquiry]: 
            list_of_indexes = st.session_state.prefix_to_columns[st.session_state.column_to_prefix[new_inquiry]]
            q_columns = st.session_state.original_column_names[list_of_indexes]
            mapping = {}
            options = []
            for col_idx, q_col in zip(list_of_indexes, q_columns):
                option = st.session_state.column_translations[q_col].split('_@_')[1]
                mapping[option] = col_idx
                options.append(option)
            try:
                int(options[0])
                answer = st.selectbox('Select your anwser', [''] + options)
                answer = [answer]
            except:
                answer = st.multiselect('Select your anwser(s)', [''] + options)
            for option in options:
                st.session_state.instance_tensor[mapping[option]] = 1 if option in answer else -1
            new_inquiry = list_of_indexes

        else:
            answer = st.selectbox('Select your anwser', [''] + ['Yes', 'No'])
            st.session_state.instance_tensor[new_inquiry] = 1 if answer == 'Yes' else -1
            new_inquiry = [new_inquiry]
        
        if st.button('Submit'): # User confirms the answer
            st.session_state.questions.append(col_english.split('_@_')[0])
            st.session_state.answers.append(answer)
            st.session_state.reveal += new_inquiry
            
            # When it needs to consider the age
            if st.session_state.question_num == st.session_state.when_to_consider_age:
                st.session_state.instance_tensor[st.session_state.age_idx] = parse_age(st.session_state.age)
                st.session_state.reveal += [st.session_state.age_idx]

            # Calculate similarity
            if st.session_state.similarity is None:
                st.session_state.similarity = torch.zeros(st.session_state.train_wo_sex_tensor.size(0), device=st.session_state.device)
            if st.session_state.question_num == st.session_state.when_to_consider_age:
                for question_idx in new_inquiry:
                    st.session_state.similarity += 2 * (st.session_state.train_wo_age_tensor[:, question_idx] != st.session_state.instance_tensor[question_idx])
                st.session_state.similarity += (st.session_state.train_wo_sex_tensor[:, st.session_state.age_idx] - st.session_state.instance_tensor[st.session_state.age_idx]).abs()
            else:
                for question_idx in new_inquiry:
                    st.session_state.similarity += 2 * (st.session_state.train_wo_age_tensor[:, question_idx] != st.session_state.instance_tensor[question_idx])
            
            st.session_state.cur_node = DPTreeNode(None, None)
            st.session_state.root.add_child(st.session_state.cur_node)
            k = st.session_state.n//10000*st.session_state.percent
            values, top_indices = torch.topk(st.session_state.similarity, k, largest=False)
            top_actual_indices = top_indices
            similar_cases = st.session_state.train_wo_age_tensor[top_actual_indices]
            selected_col = [col for col in range(st.session_state.train_wo_sex_tensor.size(1)) if col not in st.session_state.reveal]
            
            # Calculate entropy to decide the next question
            new_inquiry, en = calculate_top_entropy_columns(
                similar_cases[:, selected_col], selected_col, st.session_state.prefix_to_columns, st.session_state.column_to_prefix)
            st.session_state.cur_node.next_question = new_inquiry
            st.session_state.cur_node.entropy = np.int8(en//2)
            st.session_state.root = st.session_state.cur_node
            st.session_state.percent = st.session_state.percent//(1.4**(st.session_state.root.entropy + 1))
            st.session_state.percent = int(st.session_state.percent) if st.session_state.percent > 1 else 1
            
            if st.session_state.device == 'cuda':
                torch.cuda.synchronize()
            st.session_state.question_num += 1
            st.rerun()
            
    else: # Send user data into the predictive model
        mask_instance = pd.Series(st.session_state.instance_tensor.cpu().numpy(), index=st.session_state.original_column_names)
        mask_instance = mask_instance.astype(np.float64)
        mask_instance['AGE'] = st.session_state.age_scaler.transform(np.array(st.session_state.age).reshape(1, -1))[0][0]
        mask_instance['SEX'] = 0 if st.session_state.sex == 'Male' else 1
        X_test_tensor = torch.tensor(pd.DataFrame(mask_instance).T.values, dtype=torch.float).to(st.session_state.device)
        with torch.no_grad():
            # generate differential diagnosis
            for features in DataLoader(X_test_tensor, batch_size=32):
                features = features.unsqueeze(1) 
                outputs = st.session_state.model.forward(features)
                predictions = torch.sigmoid(outputs).round()
        
        # Print out the result
        st.markdown(f'Sex: {st.session_state.sex},{"&#160;"*10} Age: {st.session_state.age}')
        st.write('----------------')
        st.write('Agent inquiries:')
        for q, ans in zip(st.session_state.questions, st.session_state.answers):
            st.write(f'- {q}')
            if isinstance(ans, list):
                for a in ans:
                    st.markdown(f'{"&#160;"*10}{a}')
            else:
                st.markdown(f'{"&#160;"*10}{ans}')
        st.write('-----------------------')
        st.write('Predicted Differential:')
        prediction = st.session_state.all_pathologies[predictions.cpu().numpy()[0].astype(bool)]
        st.write(', '.join([st.session_state.pathology_translations[french_path] for french_path in prediction]))
        
        if st.button('Reset'):
            st.session_state.question_num = 0