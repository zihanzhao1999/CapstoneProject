import pickle
import zstandard as zstd
import torch

def read_zst(path):
    '''
    Decompress the Zstd file
    '''
    with open(path, 'rb') as compressed_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader:
            decompressed_data = reader.read()
    return pickle.loads(decompressed_data)

def parse_cols(columns_names):
    '''
    Parse the columns for the usage in the inquiry system
    '''
    prefix_to_columns = {}
    for n, col in enumerate(columns_names):
        if '_@_' in col:
            prefix, _ = col.split('_@_', 1)
            if prefix not in prefix_to_columns:
                prefix_to_columns[prefix] = []
            prefix_to_columns[prefix].append(n)
            
    column_to_prefix = {}
    for n, col in enumerate(columns_names):
        if '_@_' in col:
            prefix, _ = col.split('_@_', 1)
            column_to_prefix[n] = prefix
        else:
            column_to_prefix[n] = None
    return prefix_to_columns, column_to_prefix

def parse_age(x):
    '''
    Parse the age to several groups for the usage in the inquiry system
    '''
    if x <= 4:
        return 1
    elif x <= 15:
        return 2
    elif x <= 30:
        return 3
    elif x <= 45:
        return 4
    elif x <= 60:
        return 5
    else:
        return 6

def calculate_entropy(probabilities):
    # Ensure no zero probabilities; add a small value
    probabilities = probabilities.clamp(min=1e-9)
    return -(probabilities * probabilities.log2()).sum(dim=1)


def calculate_top_entropy_columns(df, columns, prefix_to_columns, column_to_prefix):
    '''
    Function to return the n columns with the highest entropies.
    Note that this is not completely correct 
        since it doesn't consider the correlation in the question group.
        However, for simplicity in calculation we see each column as independent.
    '''
    prob = df.float().mean(dim=0)
    prob = torch.stack([1 - prob, prob], dim=1)
    entropy_values = calculate_entropy(prob)
    index_mapping = {value: index for index, value in enumerate(columns)}
    entropy_values_group = torch.zeros(len(columns), device=df.device)
    
    maximun = 0
    max_col = None
    processed_groups = set()
    for n, col in enumerate(columns):
        if column_to_prefix[col]:
            group_prefix = column_to_prefix.get(col)
            # Skip if the group has already been calculated
            if group_prefix in processed_groups:
                continue
            indexes = [index_mapping[i] for i in prefix_to_columns[group_prefix]]
            entropy_value = entropy_values[indexes].sum()
            entropy_values_group[indexes] = entropy_value
            processed_groups.add(group_prefix)
        else:
            entropy_value = entropy_values[n]
            entropy_values_group[n] = entropy_value

    max_entropy_group_id = torch.argmax(entropy_values_group)
    max_col = columns[max_entropy_group_id]
    maximun = entropy_values_group[max_entropy_group_id].item()
    
    return (max_col, maximun)
