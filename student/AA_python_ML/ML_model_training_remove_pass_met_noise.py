import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import time
import gc
from pympler import summary, muppy
import psutil

class PylosDataset(Dataset):
    def __init__(self, csv_file: str, batch_size: int):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(csv_file, header=0, low_memory=False)
        self.BASE_Q = -5.0
        self.MARGIN = 2.0
        self.REWARD_SCALE = 1.0
        
        self.state_actions: Dict[str, List[Tuple[str, float]]] = {}
        self.states: List[str] = []
        print("Grouping moves by state...")
        for _, row in self.data.iterrows():
            state = row['current_state']
            action = row['action']
            reward = int(row['reward']) + 5
            if state not in self.state_actions:
                self.state_actions[state] = []
                self.states.append(state)
            self.state_actions[state].append((action, reward))
        print("Cleaning up original data...")
        del self.data
        gc.collect()
        print("dataset states and batches:")
        print("unique states:", len(self.states))
        print("batches:", len(self.states)/batch_size)

        print("pre-allocating tensors...")
        self.states_tensor = torch.empty((len(self.states), 30), dtype=torch.float32, device=device)
        self.validity_masks = torch.zeros((len(self.states), 304), dtype=torch.bool, device=device)  # Create before using!
        self.target_values = torch.full((len(self.states), 304), self.BASE_Q, dtype=torch.float32, device=device)
        
        # Process all states at once
        print("Converting states to tensors...")
        all_states = torch.tensor([[int(c) for c in state] for state in self.states], 
                                dtype=torch.float32, device=device)
        self.states_tensor = all_states

        print("Setting up validity masks and target values...")
        for i, state in enumerate(self.states):
            for action, reward in self.state_actions[state]:
                action_idx = action.find('1')
                self.validity_masks[i, action_idx] = True
                self.target_values[i, action_idx] = self.BASE_Q + self.MARGIN + (reward * self.REWARD_SCALE)

        # Clean up CPU memory
        del self.state_actions
        del self.states
        gc.collect()

    def __len__(self) -> int:
        return len(self.states_tensor)  # Changed from self.states to self.states_tensor
    
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
        # Simply return the pre-computed GPU tensors
        return (self.states_tensor[idx], 
                self.validity_masks[idx], 
                self.target_values[idx])
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features)
        )
        
    def forward(self, x):
        return x + self.net(x)
class LargePylosDQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Wider networks for more capacity
        self.level0_conv = nn.Sequential(
            nn.Linear(16, 256),     # 128 -> 256
            nn.LayerNorm(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)  # Added extra residual block
        )
        
        self.level1_conv = nn.Sequential(
            nn.Linear(9, 128),      # 64 -> 128
            nn.LayerNorm(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)  # Added extra residual block
        )
        
        self.level2_conv = nn.Sequential(
            nn.Linear(4, 64),       # 32 -> 64
            nn.LayerNorm(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)    # Added extra residual block
        )
        
        # Larger attention mechanism
        self.attention = nn.MultiheadAttention(1024, 16)  # 512->1024, 8->16 heads
        
        # Wider combination layers
        self.combine = nn.Sequential(
            nn.Linear(256 + 128 + 64 + 1, 1024),  # 512 -> 1024
            nn.LayerNorm(1024),
            nn.ReLU()
        )
        
        # Deeper final processing
        self.final_layers = nn.Sequential(
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),
            ResidualBlock(1024, 1024),  # Added extra residual block
            nn.Linear(1024, 304)
        )

        # Value head scaled accordingly
        self.value_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        # Process each level
        level0 = self.level0_conv(x[:, :16])
        level1 = self.level1_conv(x[:, 16:25])
        level2 = self.level2_conv(x[:, 25:29])
        level3 = x[:, 29].unsqueeze(1)
        
        # Combine features
        combined = torch.cat([level0, level1, level2, level3], dim=1)
        features = self.combine(combined)
        
        # Apply attention
        features = features.unsqueeze(0)
        attended_features, _ = self.attention(features, features, features)
        features = attended_features.squeeze(0)
        
        # Get state value and Q-values
        state_value = self.value_head(features)
        q_values = self.final_layers(features)
        
        # Advantage calculation
        advantage = q_values - q_values.mean(dim=1, keepdim=True)
        q_values = state_value + advantage
        
        return q_values
def format_action(action_idx: int) -> str:
    action_str = ['0'] * 304
    action_str[action_idx] = '1'
    return ''.join(action_str)
def export_model_to_onnx(model, save_path="pylos_model.onnx"):
    device = next(model.parameters()).device
    model.eval()
    
    # Create dummy input on same device
    dummy_input = torch.randn(1, 30, device=device)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {save_path}")
    except Exception as e:
        print(f"Export failed, trying CPU: {str(e)}")
        model = model.cpu()
        dummy_input = dummy_input.cpu()
        # Try export again
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to {save_path} after moving to CPU")


def train_dqn(data_file: str, num_epochs: int = 100, 
              batch_size: int = 512, 
              learning_rate: float = 1e-4,
              start_epoch: int = 0,
              load_path: str = None,
              escape_local_min: bool = True):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    BASE_Q = -5.0 
    MARGIN = 2.0  
    REWARD_SCALE = 1.0 
    
    # Add noise parameters
    noise_epochs = 20  # Duration of noise phase
    recovery_epochs = 30  # Duration of recovery phase
    noise_magnitude = 0.01  # Start with small noise
    original_lr = learning_rate
    
    model = LargePylosDQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataset = PylosDataset(data_file, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    # Initialize history
    history = {
        'loss': [],
        'dataset_moves_mean_q': [],
        'other_moves_mean_q': [],
        'q_value_ratio': [],
        'selection_accuracy': [],
        'best_move_accuracy': [],
        'margin_violations': []
    }

    # Initialize metrics on GPU
    metrics = {
        'loss': torch.zeros(1, device=device),
        'dataset_moves_mean_q': torch.zeros(1, device=device),
        'other_moves_mean_q': torch.zeros(1, device=device),
        'correct_selections': torch.zeros(1, device=device),
        'best_selections': torch.zeros(1, device=device),
        'margin_violations': torch.zeros(1, device=device)
    }
    
    if load_path and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device,weights_only=True)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"Resumed from epoch {start_epoch}")

    big_start_time = time.time()
    start_time = time.time()
    epoch_start_time = time.time() 

    for epoch in range(start_epoch, num_epochs):
        # Add noise phase logic
        in_noise_phase = escape_local_min and epoch < (start_epoch + noise_epochs)
        in_recovery_phase = escape_local_min and epoch < (start_epoch + noise_epochs + recovery_epochs)
        
        # Adjust learning rate based on phase
        if in_noise_phase:
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr * 2.0
        elif in_recovery_phase:
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr * 1.5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = original_lr

        # Reset metric accumulators
        for key in metrics:
            metrics[key].zero_()
        total_batches = 0
        
        for batch_idx, (states, validity_masks, target_values) in enumerate(dataloader):
            actual_batch_size = states.size(0)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                q_values = model(states)
                
                # Add noise during escape phase
                if in_noise_phase:
                    noise = torch.randn_like(q_values) * noise_magnitude
                    q_values = q_values + noise
                
                # Loss calculation on GPU
                mse_loss = nn.MSELoss()(q_values[validity_masks], target_values[validity_masks])
                # Vectorized operations for margin loss
                invalid_mask = ~validity_masks
                # Get max Q-values for invalid moves
                invalid_q = q_values.masked_fill(~invalid_mask, float('-inf'))
                other_max_q = invalid_q.max(dim=1)[0]
                # Handle case where all moves are valid
                other_max_q = torch.where(
                    invalid_mask.any(dim=1),
                    other_max_q,
                    torch.tensor(BASE_Q, device=device)
                )
                
                # Get min Q-values for valid moves
                valid_q = q_values.masked_fill(~validity_masks, float('inf'))
                dataset_min_q = valid_q.min(dim=1)[0]
                # Handle case where no moves are valid
                dataset_min_q = torch.where(
                    validity_masks.any(dim=1),
                    dataset_min_q,
                    torch.tensor(0., device=device)
                )
                margin_loss = torch.mean(torch.relu(other_max_q - dataset_min_q + MARGIN))
                loss = mse_loss + margin_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics on GPU
            with torch.no_grad():
                metrics['loss'] += loss
                metrics['dataset_moves_mean_q'] += q_values[validity_masks].mean()
                metrics['other_moves_mean_q'] += q_values[~validity_masks].mean()
                
                predicted_actions = q_values.argmax(dim=1)
                correct_selections = validity_masks[torch.arange(actual_batch_size, device=device), predicted_actions].sum()
                metrics['correct_selections'] += correct_selections
                
                best_selections = (predicted_actions == target_values.argmax(dim=1)).sum()
                metrics['best_selections'] += best_selections
                
                metrics['margin_violations'] += (other_max_q > dataset_min_q - MARGIN).sum()
            
            total_batches += 1

            # Log every 1000 batches
            if batch_idx % 1000 == 0:
                print(f"Batch {batch_idx}: {time.time() - start_time:.2f} seconds")
                start_time = time.time()

        # Transfer metrics to CPU and calculate averages
        epoch_metrics = {
            'loss': (metrics['loss'] / total_batches).item(),
            'dataset_moves_mean_q': (metrics['dataset_moves_mean_q'] / total_batches).item(),
            'other_moves_mean_q': (metrics['other_moves_mean_q'] / total_batches).item(),
            'selection_accuracy': (metrics['correct_selections'] / (total_batches * batch_size)).item(),
            'best_move_accuracy': (metrics['best_selections'] / (total_batches * batch_size)).item(),
            'margin_violations': (metrics['margin_violations'] / (total_batches * batch_size)).item()
        }
        # Calculate Q-value ratio
        epoch_metrics['q_value_ratio'] = (epoch_metrics['dataset_moves_mean_q'] / 
                                        epoch_metrics['other_moves_mean_q'] 
                                        if epoch_metrics['other_moves_mean_q'] != 0 else 0)

        # Update history
        for key in history:
            history[key].append(epoch_metrics[key])

        

        # Print epoch summary with added phase information
        print(f"\nEPOCH {epoch} time: {time.time() - epoch_start_time:.2f} seconds SUMMARY:")
        if in_noise_phase:
            print(f"[NOISE PHASE] {noise_epochs - (epoch - start_epoch)} epochs remaining")
        elif in_recovery_phase:
            print(f"[RECOVERY PHASE] {(start_epoch + noise_epochs + recovery_epochs) - epoch} epochs remaining")
        
        # Print epoch summary
        print(f"Average Loss: {epoch_metrics['loss']:.4f}")
        print(f"Average Dataset Q: {epoch_metrics['dataset_moves_mean_q']:.2f}")
        print(f"Average Other Q: {epoch_metrics['other_moves_mean_q']:.2f}")
        print(f"Selection Accuracy: {epoch_metrics['selection_accuracy']*100:.1f}%")
        print(f"Best Move Accuracy: {epoch_metrics['best_move_accuracy']*100:.1f}%")
        print(f"Margin Violations: {epoch_metrics['margin_violations']*100:.1f}%")
        print(f"Q-Value Ratio: {epoch_metrics['q_value_ratio']:.2f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print("----------------------------------------\n")

        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}_RP.pth')

    print(f"Total training time this round: {time.time() - big_start_time:.2f} seconds")
    return model, history



def plot_training_metrics(history: Dict[str, List[float]]):
    # Create a 3x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))
    
    # Plot loss
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot Q-values
    batches = range(len(history['dataset_moves_mean_q']))
    ax2.plot(batches, history['dataset_moves_mean_q'], label='Dataset Moves')
    ax2.plot(batches, history['other_moves_mean_q'], label='Other Moves')
    ax2.set_title('Average Q-Values')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Q-Value')
    ax2.legend()
    ax2.grid(True)
    
    # Plot Q-value ratio
    ax3.plot(history['q_value_ratio'])
    ax3.set_title('Q-Value Ratio (Dataset/Other)')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Ratio')
    ax3.grid(True)
    
    # Plot selection accuracies
    ax4.plot(history['selection_accuracy'], label='All Dataset Moves')
    ax4.plot(history['best_move_accuracy'], label='Best Moves')
    ax4.set_title('Selection Accuracy')
    ax4.set_xlabel('Batch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    # Plot margin violations
    ax5.plot(history['margin_violations'])
    ax5.set_title('Margin Violations')
    ax5.set_xlabel('Batch')
    ax5.set_ylabel('Violation Rate')
    ax5.grid(True)
    
    # Plot combined metrics
    ax6.plot(history['selection_accuracy'], label='Selection Acc')
    ax6.plot(history['margin_violations'], label='Margin Violations')
    ax6.plot([q/10 for q in history['q_value_ratio']], label='Q-Ratio/10')
    ax6.set_title('Combined Performance Metrics')
    ax6.set_xlabel('Batch')
    ax6.set_ylabel('Value')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final statistics
    n_batches = len(history['loss'])
    final_window = 1000  # Average over last 100 batches
    
    print("\nFinal Statistics (avg over last 100 batches):")
    print(f"Loss: {np.mean(history['loss'][-final_window:]):.4f}")
    print(f"Dataset Q-Value: {np.mean(history['dataset_moves_mean_q'][-final_window:]):.2f}")
    print(f"Other Q-Value: {np.mean(history['other_moves_mean_q'][-final_window:]):.2f}")
    print(f"Q-Value Ratio: {np.mean(history['q_value_ratio'][-final_window:]):.2f}")
    print(f"Selection Accuracy: {np.mean(history['selection_accuracy'][-final_window:])*100:.1f}%")
    print(f"Best Move Accuracy: {np.mean(history['best_move_accuracy'][-final_window:])*100:.1f}%")
    print(f"Margin Violations: {np.mean(history['margin_violations'][-final_window:])*100:.1f}%")


def cleanup_between_datasets():
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Print memory stats to verify cleanup
    print("\nMemory status after cleanup:")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    process = psutil.Process()
    print(f"System memory usage: {process.memory_info().rss / 1024**2:.2f} MB")
    print("----------------------------------------\n")
    

if __name__ == "__main__":
    start_epoch = 471  # Your current epoch
    num_epochs = 601  # Your target epoch
    load_path = "checkpoint_epoch_470_RP.pth"  # Your current checkpoint

    model, history = train_dqn("data_shuffled_remove_pass.csv", 
                        num_epochs=num_epochs,
                        start_epoch=start_epoch,
                        load_path=load_path,
                        escape_local_min=True)  # Enable noise escape
    plot_training_metrics(history)
    cleanup_between_datasets()
    export_model_to_onnx(model, "pylos_model_remove_pass_final_noise.onnx")

