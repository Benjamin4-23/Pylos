import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import gc

def create_data_subsets(input_file: str, output_files: List[str]):
    total_splits = len(output_files)
    remaining_data = pd.read_csv(input_file)
    states_per_split = (remaining_data['current_state'].nunique() // total_splits)+1

    for i in range(total_splits):
        print(f"Creating subset {i+1} with {states_per_split} unique states...")
        
        # Initialize counters
        unique_states = set()
        rows_to_keep = 0
        
        # Iterate over the remaining data
        for idx, row in remaining_data.iterrows():
            state = row['current_state']
            
            if len(unique_states) == states_per_split:
                break
                
            unique_states.add(state)
            rows_to_keep += 1

        # Save the current subset
        subset_df = remaining_data.iloc[:rows_to_keep]
        subset_df.to_csv(output_files[i], index=False)
        
        # Remove used rows
        remaining_data = remaining_data.iloc[rows_to_keep:]
        
        # Print statistics
        print(f"Number of rows in subset {i+1}: {len(subset_df)}")
        print(f"Number of unique states in subset {i+1}: {subset_df['current_state'].nunique()}")
        print(f"Average actions per state in subset {i+1}: {len(subset_df)/len(unique_states):.2f}")
        
        # Cleanup
        del subset_df
        gc.collect()
    
    # Save any remaining data to the last file
    if not remaining_data.empty:
        print("Saving remaining data to the last file...")
        print(f"Number of rows in the last file: {len(remaining_data)}")
        print(f"Number of unique states in the last file: {remaining_data['current_state'].nunique()}")
        del remaining_data
        gc.collect()

if __name__ == "__main__":
    create_data_subsets(
        input_file="data_shuffled_remove_pass.csv",
        output_files=["data_subset_1.csv", "data_subset_2.csv", "data_subset_3.csv"]
    ) 