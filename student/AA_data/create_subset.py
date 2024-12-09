import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import gc

def create_data_subset(input_file: str, output_file: str, n_unique_states: int = 500000):
    print("Creating subset with exact number of unique states...")
    
    # Initialize counters
    unique_states = set()
    rows_to_keep = 0
    
    # Read in chunks to find the cutoff point
    print("Finding cutoff point...")
    with pd.read_csv(input_file, chunksize=100000) as reader:
        for chunk in reader:
            for idx, row in chunk.iterrows():
                state = row['current_state']
                unique_states.add(state)
                rows_to_keep += 1
                
                if len(unique_states) == n_unique_states:
                    print(f"Found {n_unique_states} unique states after {rows_to_keep} rows")
                    break
            if len(unique_states) == n_unique_states:
                break
    
    # Read the exact number of rows needed
    print(f"Reading first {rows_to_keep} rows...")
    subset_df = pd.read_csv(input_file, nrows=rows_to_keep)
    
    # Save subset
    print("Saving subset to file...")
    subset_df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Number of rows in subset: {len(subset_df)}")
    print(f"Number of unique states in subset: {subset_df['current_state'].nunique()}")
    print(f"Average actions per state: {len(subset_df)/len(unique_states):.2f}")
    
    # Cleanup
    del subset_df
    gc.collect()

if __name__ == "__main__":
    create_data_subset(
        input_file="data_shuffled.csv",
        output_file="data_subset_1k.csv",
        n_unique_states=1024
    ) 