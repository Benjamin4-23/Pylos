import pandas as pd
import glob
import sys
from pathlib import Path
import numpy as np

def read_data_file(file_path):
    try:
        # Read CSV with specific column names and datatypes
        df = pd.read_csv(
            file_path, 
            dtype={
                'current_state': str,
                'action': str,
                'reward': float
            }
        )
        print(f"Successfully read {file_path} with {len(df)} rows")
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: {file_path} is empty", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}", file=sys.stderr)
        return pd.DataFrame()

def main():
    data_files = glob.glob("data_part*.csv")
    
    if not data_files:
        print("No data_part_*.csv files found in current directory", file=sys.stderr)
        return

    dfs = []
    for file_path in data_files:
        df = read_data_file(file_path)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print("No valid data found in any files", file=sys.stderr)
        return

    print(f"Combining {len(dfs)} files")
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Combined {len(combined_df)} rows, starting removal")
    # Count total rows before removal
    total_rows_before = len(combined_df)
    
    # Find the index of '1' in the 'action' string and remove rows where index < 273 (move and place actions)
    combined_df['action_index'] = combined_df['action'].apply(lambda x: x.index('1'))
    remove_pass_mask = combined_df['action_index'] < 273
    removed_rows = combined_df[remove_pass_mask]
    combined_df = combined_df[~remove_pass_mask]
    print("remove indexes done")
    
    # Count removed actions
    removed_remove_actions = sum((removed_rows['action_index'] >= 273) & (removed_rows['action_index'] <= 302))
    removed_pass_actions = sum(removed_rows['action_index'] == 303)
    print("removed actions counting done, starting shuffling")

    # Faster state-based shuffling
    # Get the state boundaries
    state_changes = combined_df['current_state'].ne(combined_df['current_state'].shift()).cumsum()
    unique_states = state_changes.unique()
    print("unique states done")
    
    # Shuffle the state indices
    np.random.shuffle(unique_states)
    print("shuffled states done")
    
    # Create a mapping from old to new positions
    position_map = dict(zip(unique_states, range(len(unique_states))))
    print("position map done")
    
    # Apply the mapping to reorder the groups
    combined_df['shuffle_key'] = state_changes.map(position_map)
    print("shuffled groups done")
    shuffled_df = combined_df.sort_values('shuffle_key').drop('shuffle_key', axis=1)
    print("sorted groups done")
    
    # Remove the temporary action_index column
    shuffled_df = shuffled_df.drop('action_index', axis=1)
    print("dropped action index done")
    # Save to CSV
    shuffled_df.to_csv("data_shuffled.csv", index=False)
    print("saved to csv done")
    # Print summary
    print(f"\nSummary:")
    print(f"Processed {len(data_files)} files")
    print(f"Total rows before removal: {total_rows_before}")
    print(f"Removed REMOVE actions: {removed_remove_actions}")
    print(f"Removed PASS actions: {removed_pass_actions}")
    print(f"Total rows after removal: {len(combined_df)}")
    print(f"Unique states: {len(combined_df['current_state'].unique())}")
    print(f"Output saved to: data_shuffled.csv")

if __name__ == "__main__":
    main()
