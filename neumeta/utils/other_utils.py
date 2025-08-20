import torch
import numpy as np
import random
from prettytable import PrettyTable
from omegaconf import OmegaConf
import argparse
import torch.nn as nn
import numpy as np
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a NeRF model with CIFAR-10")

    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Ratio used for training purposes')
    parser.add_argument('--resume_from', type=str,
                        help='Checkpoint file path to resume training from')
    parser.add_argument('--load_from', type=str,
                        help='Checkpoint file path to load')
    parser.add_argument('--test_result_path', type=str,
                        help='Path to save the test result')
    parser.add_argument('--test', action='store_true',
                        default=False, help='Test the model')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Load the base configuration
    if config.get('base_config', None):
        print("Loading base config from " + config.base_config)
        base_config = OmegaConf.load(config.base_config)
        config = OmegaConf.merge(base_config, config)

    # Convert args to a dictionary
    # We filter out None values and the 'config' argument
    cli_args = {k: v for k, v in vars(args).items()}

    # Merge command-line arguments into the configuration
    config = OmegaConf.merge(config, cli_args)
    if len(config.dimensions.range) == 2:
        interval = config.dimensions.get('interval', 1)
        config.dimensions.range = list(
            range(config.dimensions.range[0], config.dimensions.range[1] + 1, interval))
    return config


def print_omegaconf(cfg):
    """
    Print an OmegaConf configuration in a table format.

    :param cfg: OmegaConf configuration object.
    """
    # Flatten the OmegaConf configuration to a dictionary
    flat_config = OmegaConf.to_container(cfg, resolve=True)

    # Create a table with PrettyTable
    table = PrettyTable()

    # Define the column names
    table.field_names = ["Key", "Value"]

    # Recursively go through the items and add rows
    def add_items(items, parent_key=""):
        for k, v in items.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is another dict, recursively add its items
                add_items(v, parent_key=current_key)
            else:
                # If it's a leaf node, add it to the table
                table.add_row([current_key, v])

    # Start adding items from the top-level configuration
    add_items(flat_config)

    # Print the table
    print(table)


def set_seed(seed_value=42):
    """Set the seed for generating random numbers for PyTorch and other libraries to ensure reproducibility.

    Args:
        seed_value (int, optional): The seed value. Defaults to 42 (a commonly used value in randomized algorithms requiring a seed).
    """
    print("Setting seed..." + str(seed_value) + " for reproducibility")
    # Set the seed for generating random numbers in Python's random library.
    random.seed(seed_value)

    # Set the seed for generating random numbers in NumPy, which can also affect randomness in cases where PyTorch relies on NumPy.
    np.random.seed(seed_value)

    # Set the seed for generating random numbers in PyTorch. This affects the randomness of various PyTorch functions and classes.
    torch.manual_seed(seed_value)

    # If you are using CUDA, and want to generate random numbers on the GPU, you need to set the seed for CUDA as well.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        # For multi-GPU, if you are using more than one GPU.
        torch.cuda.manual_seed_all(seed_value)

        # Additionally, for even more deterministic behavior, you might need to set the following environment, though it may slow down the performance.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        self.set_shadow(model)

    def set_shadow(self, model):
        # Initialize the shadow weights with the model's weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def apply(self):
        # Backup the current model weights and set the model's weights to the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        # Restore the original model weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def update(self):
        # Update the shadow weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * \
                    self.shadow[name] + (1.0 - self.decay) * param.data


def save_checkpoint(filepath, model, optimizer, ema, epoch, best_acc):
    """
    Saves the current state including a model, optimizer, and EMA shadow weights.

    Args:
    filepath (str): The file path where the checkpoint will be saved.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    epoch (int): The current epoch.
    best_acc (float): The best accuracy observed during training.
    """
    # Save the model, optimizer, EMA shadow weights, and other elements
    if ema is not None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow,  # specifically saving shadow weights
            'best_acc': best_acc,
        }
    else:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
        }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, ema, device='cuda'):
    """
    Loads the state from a checkpoint into the model, optimizer, and EMA object.

    Args:
    filepath (str): The file path to load the checkpoint from.
    model (torch.nn.Module): The model.
    optimizer (torch.optim.Optimizer): The optimizer.
    ema (EMA): The EMA object.
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if ema is not None:
        ema.shadow = {k: checkpoint['ema_shadow'][k].to(
            device) for k in checkpoint['ema_shadow']}
    # ema.shadow = {k:checkpoint['ema_shadow'][k].to(device) for k in checkpoint['ema_shadow'] }  # specifically loading shadow weights

    return checkpoint  # Contains other information like epoch, best_acc

# Generates all 2^6 = 64 combinations of 0s and 1s for 6 dimensions
BOX_OFFSETS_6D = torch.tensor([
    [i, j, k, l, m, n]
    for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1] for m in [0, 1] for n in [0, 1]
], dtype=torch.int32)

def hash_6d(coords, log2_hashmap_size):
    '''
    Hash function for 6D coordinates
    coords: B x 64 x 6 (64 vertices of 6D hypercube)
    '''
    # Large prime numbers for 6D hashing (extending the 5D approach)
    # Added a 6th large prime to the list. These are typically chosen to be
    # unique and large to help distribute hashes evenly and reduce collisions.
    primes = torch.tensor([1, 2654435761, 805459861, 3674653429, 2097192037, 4294967291], # Added 6th prime
                          device=coords.device, dtype=torch.long)
    
    # Initialize XOR result tensor with the correct shape (batch_size x num_corners)
    # coords.shape[:-1] will give (B, 64)
    xor_result = torch.zeros(coords.shape[:-1], dtype=torch.long, device=coords.device)
    
    # Iterate through all 6 dimensions
    for i in range(6): # Changed range from 5 to 6
        # Apply XOR and multiply each coordinate by its corresponding prime
        # coords[..., i] accesses the i-th dimension for all batch items and all 64 corners
        xor_result ^= coords[..., i].long() * primes[i]
    
    # Apply modulo operation to ensure the hash fits within the hashmap size
    return xor_result % (2**log2_hashmap_size)

# --- get_voxel_vertices_6d function ---
def get_voxel_vertices_6d(x, bounding_box_6d, resolution, log2_hashmap_size):
    '''
    x: 6D coordinates of samples. B x 6
    bounding_box_6d: min and max 6D coordinates for the model space. (min_vals_6d, max_vals_6d)
    resolution: 6D tensor, number of voxels per axis for current level
    '''
    min_box, max_box = bounding_box_6d
    
    is_within_bounds_per_dim = (x >= min_box) & (x <= max_box)
    keep_mask = is_within_bounds_per_dim.all(dim=-1) # Shape B
    
    x_clamped = torch.clamp(x, min=min_box, max=max_box)
    
    grid_size = (max_box - min_box) / (resolution + 1e-6) # 6D tensor

    bottom_left_idx = torch.floor((x_clamped - min_box) / grid_size).int() # B x 6

    voxel_min_vertex = bottom_left_idx.float() * grid_size + min_box # B x 6
    voxel_max_vertex = voxel_min_vertex + grid_size # B x 6

    # Generate indices for all 2^6 = 64 vertices of the 6D hypercube
    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS_6D.to(x.device) # B x 64 x 6

    # Hash these 6D indices
    hashed_voxel_indices = hash_6d(voxel_indices, log2_hashmap_size) # B x 64

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask

def get_bbox5d_for_model_collection(models, padding_factor=0.1):
    """
    Get 5D bounding box from a collection of neural network models
    Similar to how NeRF gets bbox from camera poses
    
    Args:
        models: List of PyTorch nn.Module models
        padding_factor: Extra padding around the bounds (default 10%)
    
    Returns:
        bounding_box_5d: (box_min, box_max) tensors of shape [5]
    """
    min_bound = [float('inf')] * 5
    max_bound = [float('-inf')] * 5
    
    all_configs = []
    
    for model_idx, model in enumerate(models):
        layer_id = 0
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d, nn.Conv3d)):
                
                if isinstance(module, nn.Linear):
                    in_features = module.in_features
                    out_features = module.out_features
                    # For linear layers: [layer_id, 0, 0, in_features, out_features]
                    config = [layer_id, 0, 0, in_features, out_features]
                    
                elif isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    
                    # Calculate effective size based on parameters
                    if hasattr(module, 'kernel_size'):
                        kernel_size = module.kernel_size
                        if isinstance(kernel_size, (tuple, list)):
                            kernel_prod = np.prod(kernel_size)
                        else:
                            kernel_prod = kernel_size
                    else:
                        kernel_prod = 1
                    
                    in_size = in_channels * kernel_prod
                    out_size = out_channels * kernel_prod
                    
                    config = [layer_id, in_channels, out_channels, in_size, out_size]
                
                all_configs.append(config)
                
                # Update bounds
                def find_min_max(cfg):
                    for i in range(5):
                        if min_bound[i] > cfg[i]:
                            min_bound[i] = cfg[i]
                        if max_bound[i] < cfg[i]:
                            max_bound[i] = cfg[i]
                
                find_min_max(config)
                layer_id += 1
    
    # Convert to tensors and add padding
    box_min = torch.tensor(min_bound, dtype=torch.float32)
    box_max = torch.tensor(max_bound, dtype=torch.float32)
    
    # Add padding (similar to NeRF padding)
    padding = (box_max - box_min) * padding_factor
    box_min = box_min - padding
    box_max = box_max + padding
    
    # Ensure minimum bounds are non-negative where appropriate
    box_min[0] = max(0.0, box_min[0])  # layer_id >= 0
    box_min[1] = max(0.0, box_min[1])  # in_channels >= 0  
    box_min[2] = max(0.0, box_min[2])  # out_channels >= 0
    box_min[3] = max(1.0, box_min[3])  # in_size >= 1
    box_min[4] = max(1.0, box_min[4])  # out_size >= 1
    
    return (box_min, box_max)

def get_bbox5d_for_dataset(dataloader, sample_fraction=1.0):
    """
    Get 5D bounding box by sampling from your dataset
    Similar to how NeRF samples rays from different camera positions
    
    Args:
        dataloader: DataLoader containing 5D samples
        sample_fraction: Fraction of dataset to sample (for efficiency)
    
    Returns:
        bounding_box_5d: (box_min, box_max) tensors
    """
    min_bound = [float('inf')] * 5
    max_bound = [float('-inf')] * 5
    
    total_samples = 0
    samples_to_process = int(len(dataloader) * sample_fraction)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= samples_to_process:
            break
            
        # Extract 5D coordinates from batch
        # Adjust this based on your data format
        if isinstance(batch, (list, tuple)):
            coords_5d = batch[0]  # Assuming first element contains coordinates
        else:
            coords_5d = batch
            
        # Ensure it's the right shape [B, 5]
        if coords_5d.dim() == 1:
            coords_5d = coords_5d.unsqueeze(0)
        
        # Update bounds for each sample in batch
        batch_min = torch.min(coords_5d, dim=0)[0]
        batch_max = torch.max(coords_5d, dim=0)[0]
        
        for i in range(5):
            if min_bound[i] > batch_min[i].item():
                min_bound[i] = batch_min[i].item()
            if max_bound[i] < batch_max[i].item():
                max_bound[i] = batch_max[i].item()
        
        total_samples += coords_5d.shape[0]
    
    print(f"Processed {total_samples} samples from {batch_idx+1} batches")
    
    # Convert to tensors and add small padding
    box_min = torch.tensor(min_bound, dtype=torch.float32)
    box_max = torch.tensor(max_bound, dtype=torch.float32)
    
    # Add 5% padding (smaller than model collection since data is more precise)
    padding = (box_max - box_min) * 0.05
    box_min = box_min - padding
    box_max = box_max + padding
    
    return (box_min, box_max)

def get_bbox5d_for_architecture_family(arch_configs, padding_factor=0.15):
    """
    Get 5D bounding box for a family of architectures (e.g., ResNet variants)
    
    Args:
        arch_configs: List of architecture specifications
                     Each config: dict with 'layers', 'channels', 'sizes' etc.
        padding_factor: Extra padding for generalization
    
    Returns:
        bounding_box_5d: (box_min, box_max) tensors
    """
    min_bound = [float('inf')] * 5
    max_bound = [float('-inf')] * 5
    
    for config in arch_configs:
        # Extract architecture parameters
        max_layers = config.get('num_layers', 50)
        channel_configs = config.get('channels', [64, 128, 256, 512])
        size_configs = config.get('sizes', [224*224, 112*112, 56*56, 28*28])
        
        # Generate representative configurations for this architecture
        for layer_id in range(max_layers):
            for in_ch in channel_configs:
                for out_ch in channel_configs:
                    for in_size in size_configs:
                        for out_size in size_configs:
                            point = [layer_id, in_ch, out_ch, in_size, out_size]
                            
                            def find_min_max(pt):
                                for i in range(5):
                                    if min_bound[i] > pt[i]:
                                        min_bound[i] = pt[i]
                                    if max_bound[i] < pt[i]:
                                        max_bound[i] = pt[i]
                            
                            find_min_max(point)
    
    # Convert and add padding
    box_min = torch.tensor(min_bound, dtype=torch.float32)
    box_max = torch.tensor(max_bound, dtype=torch.float32)
    
    padding = (box_max - box_min) * padding_factor
    box_min = box_min - padding
    box_max = box_max + padding
    
    # Ensure valid bounds
    box_min = torch.clamp(box_min, min=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    
    return (box_min, box_max)

def get_bbox5d_for_metamorphosis_space(source_models, target_models, interpolation_steps=10):
    """
    Get 5D bounding box for neural metamorphosis interpolation space
    Considers the path between source and target architectures
    
    Args:
        source_models: List of source architecture models
        target_models: List of target architecture models  
        interpolation_steps: Number of interpolation steps to consider
    
    Returns:
        bounding_box_5d: (box_min, box_max) tensors
    """
    min_bound = [float('inf')] * 5
    max_bound = [float('-inf')] * 5
    
    def extract_model_configs(model):
        configs = []
        layer_id = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                if isinstance(module, nn.Linear):
                    config = [layer_id, 0, 0, module.in_features, module.out_features]
                elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                    in_ch, out_ch = module.in_channels, module.out_channels
                    kernel_size = np.prod(module.kernel_size) if hasattr(module, 'kernel_size') else 1
                    config = [layer_id, in_ch, out_ch, in_ch * kernel_size, out_ch * kernel_size]
                configs.append(config)
                layer_id += 1
        return configs
    
    # Process source models
    for model in source_models:
        configs = extract_model_configs(model)  
        for config in configs:
            for i in range(5):
                min_bound[i] = min(min_bound[i], config[i])
                max_bound[i] = max(max_bound[i], config[i])
    
    # Process target models
    for model in target_models:
        configs = extract_model_configs(model)
        for config in configs:
            for i in range(5):
                min_bound[i] = min(min_bound[i], config[i])
                max_bound[i] = max(max_bound[i], config[i])
    
    # Consider interpolation paths (linear interpolation in parameter space)
    if len(source_models) > 0 and len(target_models) > 0:
        source_configs = extract_model_configs(source_models[0])
        target_configs = extract_model_configs(target_models[0])
        
        min_len = min(len(source_configs), len(target_configs))
        for i in range(min_len):
            src_config = source_configs[i]
            tgt_config = target_configs[i]
            
            # Sample interpolation path
            for t in np.linspace(0, 1, interpolation_steps):
                interp_config = [
                    src_config[j] * (1-t) + tgt_config[j] * t 
                    for j in range(5)
                ]
                
                for j in range(5):
                    min_bound[j] = min(min_bound[j], interp_config[j])
                    max_bound[j] = max(max_bound[j], interp_config[j])
    
    # Convert and add metamorphosis-specific padding
    box_min = torch.tensor(min_bound, dtype=torch.float32)
    box_max = torch.tensor(max_bound, dtype=torch.float32)
    
    # Larger padding for metamorphosis to handle interpolation artifacts
    padding = (box_max - box_min) * 0.2
    box_min = box_min - padding
    box_max = box_max + padding
    
    # Ensure valid bounds
    box_min[0] = max(0.0, box_min[0])  # layer_id >= 0
    box_min[1] = max(0.0, box_min[1])  # channels >= 0
    box_min[2] = max(0.0, box_min[2])  
    box_min[3] = max(1.0, box_min[3])  # sizes >= 1
    box_min[4] = max(1.0, box_min[4])
    
    return (box_min, box_max)