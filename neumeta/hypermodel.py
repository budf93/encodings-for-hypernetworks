import torch.nn as nn
import math
from math import pi
import numpy as np
import torch
import torch.nn.functional as F
from .utils.other_utils import get_voxel_vertices_6d

def weights_init_uniform_relu(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) 
        if m.bias is not None: 
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight) 
            bound = 1 / math.sqrt(fan_in) 
            torch.nn.init.uniform_(m.bias, -bound, bound) 


class LearnablePositionalEmbeddings(nn.Module):
    def __init__(self, d_model, seq_len):
        super(LearnablePositionalEmbeddings, self).__init__()
        
        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        # Add the learnable positional embeddings to the input
        return x + self.positional_embeddings

    
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, input_dim=5):
        super(PositionalEncoding, self).__init__()
        print("num_freqs: ", num_freqs, type(num_freqs))
        if isinstance(num_freqs, int):
            self.freqs_low = 0
            self.freqs_high = num_freqs
        elif len(num_freqs) == 2:
            self.freqs_low = num_freqs[0]
            self.freqs_high = num_freqs[1]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
            
        self.input_dim = input_dim
        # print(f'input_dim : {input_dim}')

    def forward(self, x):
        # x: [B, 5] (layer_id, in_channel_id, out_channel_id, in_layer_size, out_layer_size)
        #input_dim should be 6 not 5
        # print(f'input_dim : {self.input_dim}')
        # exit()
        out = [x]
        for i in range(self.freqs_low, self.freqs_high):
            freq = 2.0**i * np.pi
            for j in range(self.input_dim):  # 5 input dimensions
                out.append(torch.sin(freq * x[:, j].unsqueeze(-1)))
                out.append(torch.cos(freq * x[:, j].unsqueeze(-1)))
        return torch.cat(out, dim=-1)  # [B, 5 + 10*5]
    
# Generates all 2^6 = 64 combinations of 0s and 1s for 6 dimensions
BOX_OFFSETS_6D = torch.tensor([
    [i, j, k, l, m, n]
    for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1] for m in [0, 1] for n in [0, 1]
], dtype=torch.int32)

#taken from https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/hash_encoding.py
class HashEmbedder6D(nn.Module):
    def __init__(self, bounding_box_6d, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super(HashEmbedder6D, self).__init__()
        
        # Bounding box should be a tuple of (min_coords, max_coords) where each is a 6D tensor
        if not (isinstance(bounding_box_6d, tuple) and len(bounding_box_6d) == 2 and
                bounding_box_6d[0].shape == (6,) and bounding_box_6d[1].shape == (6,)):
            raise ValueError("bounding_box_6d must be a tuple (min_coords, max_coords) of two 6D tensors.")
        self.bounding_box_6d = bounding_box_6d 
        
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.out_dim = self.n_levels * self.n_features_per_level

        # Allow base_resolution and finest_resolution to be 6D tensors for per-dimension control
        if isinstance(base_resolution, (int, float)):
            self.base_resolution = torch.full((6,), float(base_resolution), dtype=torch.float32)
        elif isinstance(base_resolution, (list, tuple)) and len(base_resolution) == 6:
            self.base_resolution = torch.tensor(base_resolution, dtype=torch.float32)
        else:
            raise ValueError("base_resolution must be an int/float or a list/tuple of length 6.")

        if isinstance(finest_resolution, (int, float)):
            self.finest_resolution = torch.full((6,), float(finest_resolution), dtype=torch.float32)
        elif isinstance(finest_resolution, (list, tuple)) and len(finest_resolution) == 6:
            self.finest_resolution = torch.tensor(finest_resolution, dtype=torch.float32)
        else:
            raise ValueError("finest_resolution must be an int/float or a list/tuple of length 6.")

        # self.b is now a 6D tensor
        self.b = torch.exp((torch.log(self.finest_resolution) - torch.log(self.base_resolution)) / (n_levels - 1))

        self.embeddings = nn.ModuleList([nn.Embedding(2**self.log2_hashmap_size,
                                        self.n_features_per_level) for _ in range(n_levels)])
        for i in range(n_levels):
            nn.init.uniform_(self.embeddings[i].weight, a=-0.0001, b=0.0001)

    def multilinear_interp_6d(self, x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
        '''
        6D multilinear interpolation
        x: B x 6
        voxel_min_vertex: B x 6
        voxel_max_vertex: B x 6
        voxel_embedds: B x 64 x n_features_per_level  (2^6 = 64 vertices for 6D hypercube)
        '''
        weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex + 1e-6) # B x 6
        weights = torch.clamp(weights, 0.0, 1.0)

        result = torch.zeros(x.shape[0], self.n_features_per_level, device=x.device)
        
        # Iterate through all 2^6 = 64 vertices
        for i in range(64):
            # Get the binary representation of 'i' to determine which corner it is
            vertex_binary = [(i >> d) & 1 for d in range(6)] # [b0, b1, b2, b3, b4, b5]
            
            # Calculate interpolation weight for this specific vertex
            weight = torch.ones(x.shape[0], device=x.device)
            for dim in range(6):
                if vertex_binary[dim] == 0:
                    weight *= (1 - weights[:, dim])
                else: # vertex_binary[dim] == 1
                    weight *= weights[:, dim]
            
            result += weight.unsqueeze(-1) * voxel_embedds[:, i, :]
        
        return result

    def forward(self, x):
        # x is 6D point position: B x 6 (l/L', cin/Cin, cout/Cout, L/N, Cin/N, Cout/N)
        x_embedded_all = []
        # Ensure bounding_box_6d is on the correct device when accessed
        local_bounding_box_6d = (self.bounding_box_6d[0].to(x.device), self.bounding_box_6d[1].to(x.device))
        
        # Ensure base_resolution, finest_resolution, and b are on the correct device
        local_base_resolution = self.base_resolution.to(x.device)
        local_finest_resolution = self.finest_resolution.to(x.device)
        local_b = self.b.to(x.device)

        overall_keep_mask = None # Initialize to None for first level
        for i in range(self.n_levels):
            resolution_per_dim = torch.floor(local_base_resolution * (local_b**i))
            resolution_per_dim = torch.max(resolution_per_dim, torch.tensor(1.0, device=x.device))

            voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices, keep_mask_level = get_voxel_vertices_6d(
                                                x, local_bounding_box_6d, 
                                                resolution_per_dim, self.log2_hashmap_size)
            
            voxel_embedds = self.embeddings[i](hashed_voxel_indices) # B x 64 x n_features_per_level

            x_embedded = self.multilinear_interp_6d(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds)
            x_embedded_all.append(x_embedded)

            if overall_keep_mask is None:
                overall_keep_mask = keep_mask_level.all(dim=-1)
            else:
                overall_keep_mask = overall_keep_mask & keep_mask_level.all(dim=-1)
        
        if not x_embedded_all:
             return torch.empty(x.shape[0], self.out_dim, device=x.device), torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

        return torch.cat(x_embedded_all, dim=-1), overall_keep_mask
    
#taken from https://github.com/ndahlquist/pytorch-fourier-feature-networks
class GaussianFourierFeatureTransform(nn.Module):
    """
    An adjusted implementation of Gaussian Fourier feature mapping to mimic the behavior
    of PositionalEncoding from the provided code. It applies a Gaussian random projection
    followed by sine and cosine transformations to each input dimension.

    Based on: "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"
    (https://arxiv.org/abs/2006.10739, https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html)

    Input: Tensor of shape [B, input_dim]
    Output: Tensor of shape [B, input_dim + 2 * input_dim * num_freqs]
    """

    #input_dim should be 6 not 5
    def __init__(self, num_freqs=10, input_dim=5, scale=10):
        """
        Args:
            num_freqs (int or tuple): Number of frequency bands or a tuple (freqs_low, freqs_high).
            input_dim (int): Number of input channels/dimensions (e.g., 5).
            scale (float): Standard deviation for Gaussian sampling of projection matrix B.
        """
        super(GaussianFourierFeatureTransform, self).__init__()

        # Handle num_freqs as int or tuple, similar to PositionalEncoding
        if isinstance(num_freqs, int):
            self.freqs_low = 0
            self.freqs_high = num_freqs
        elif isinstance(num_freqs, (list, tuple)) and len(num_freqs) == 2:
            self.freqs_low = num_freqs[0]
            self.freqs_high = num_freqs[1]
        else:
            raise ValueError("num_freqs should be either an integer or a list/tuple of length 2.")

        self.input_dim = input_dim
        self.num_freqs = self.freqs_high - self.freqs_low
        self.scale = scale

        # Initialize Gaussian projection matrix B for each frequency band
        # Shape: [input_dim, num_freqs], where each entry is sampled from N(0, scale^2)
        self._B = nn.Parameter(
            torch.randn(input_dim, self.num_freqs) * scale, requires_grad=False
        )
        # print(f'input_dim : {input_dim}')

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [B, input_dim + 2 * input_dim * num_freqs]
        """
        # Check input dimensions
        assert x.dim() == 2, f"Expected 2D input (got {x.dim()}D input)"
        assert x.shape[1] == self.input_dim, \
            f"Expected input to have {self.input_dim} channels (got {x.shape[1]} channels)"
        
        # print(f'input_dim : {self.input_dim}')
        # exit()
        # Initialize output list with the original input
        out = [x]

        # Apply Gaussian Fourier features for each input dimension
        for j in range(self.input_dim):
            # Extract the j-th dimension: [B, 1]
            x_j = x[:, j].unsqueeze(-1)

            # Project using the j-th row of B: [B, 1] @ [1, num_freqs] -> [B, num_freqs]
            proj = x_j @ self._B[j:j+1, :].to(x.device)

            # Apply 2π scaling and sine/cosine transformations
            proj = 2 * np.pi * proj  # [B, num_freqs]
            out.append(torch.sin(proj))  # [B, num_freqs]
            out.append(torch.cos(proj))  # [B, num_freqs]

        # Concatenate all outputs along the last dimension
        # Output shape: [B, input_dim + 2 * input_dim * num_freqs]
        return torch.cat(out, dim=-1)

class NeRF_MLP_Residual_Scaled(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=None, num_layers=4, scalar=0.0):
        """
        A NeRF MLP with residual connections and scaled activations.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layers.
            output_dim (int): Dimension of the output.
            num_layers (int): Number of hidden layers.
            scalar (float): Initial value for the learnable scalar for scaling residuals.
        """
        super(NeRF_MLP_Residual_Scaled, self).__init__()

        # Define the initial layer
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        
        # Create ModuleList for residual layers and scalars
        self.residual_blocks = nn.ModuleList()
        self.scalars = nn.ParameterList()

        # Adding residual blocks and their corresponding scalars
        for _ in range(num_layers - 1):
            self.residual_blocks.append(nn.Linear(hidden_dim, hidden_dim))
            self.scalars.append(nn.Parameter(torch.tensor(scalar), requires_grad=True))
        
        # Activation function
        self.act = nn.ReLU(inplace=True)
        
        # Define the output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the network.
        """
        # Initial transformation
        # print('error here hypermodel 295')
        x = self.act(self.initial_layer(x))
        
        # Process through each residual block
        for block, scale in zip(self.residual_blocks, self.scalars):
            residual = x  # Store the residual
            out = block(x)
            x = scale * self.act(out) + residual  # Apply scaled activation and add residual

        # Final transformation
        x = self.output_layer(x)
        return x


class NeRF_MLP_Compose_PE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
        """
        NeRF_MLP_Compose_PE is a class that represents a composition of NeRF_MLP_Residual_Scaled models.

        Args:
            input_dim (int): The input dimension of the model.
            hidden_dim (int): The hidden dimension of the model.
            output_dim (int): The output dimension of the model.
            num_freqs (int or list of length 2, optional): The number of frequencies used for positional encoding. Defaults to 10.
            num_layers (int, optional): The number of layers in each NeRF_MLP_Residual_Scaled model. Defaults to 4.
            num_compose (int, optional): The number of NeRF_MLP_Residual_Scaled models to compose. Defaults to 4.
            normalizing_factor (float, optional): The normalizing factor for layer_id and input_dim. Defaults to 1.0.
        """
        super(NeRF_MLP_Compose_PE, self).__init__()

        self.positional_encoding = PositionalEncoding(num_freqs, input_dim=input_dim)
        self.output_dim = output_dim
        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        if isinstance(num_freqs, int):
            num_freqs = num_freqs
        elif len(num_freqs) == 2:
            num_freqs = num_freqs[1] - num_freqs[0]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
            
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers))
            
        self.apply(weights_init_uniform_relu)
            
    def forward(self, x, layer_id=None, input_dim=None):
        """
        Forward pass of the NeRF_MLP_Compose_PE model.

        Args:
            x (torch.Tensor): The input tensor.
            layer_id (torch.Tensor, optional): The layer ID tensor. Defaults to None.
            input_dim (torch.Tensor, optional): The input dimension tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        if layer_id is None:
            layer_id = (x[:, 0] * self.norm).int()
        if input_dim is None:
            input_dim = (x[:, -1] * self.norm)
        x[:, :3] = x[:, :3] / x[:, 3:]
        x = self.positional_encoding(x)
        unique_layer_ids = torch.unique(layer_id)
        output_x = torch.zeros((x.size(0), self.output_dim)).to(x.device)
        for lid in unique_layer_ids:
            mask = lid == layer_id
            output_x[mask] = self.model[lid].forward(x[mask])   
        return output_x / (input_dim.unsqueeze(-1))
    
class NeRF_MLP_Compose_NoEncodings(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
        """
        NeRF_MLP_Compose_PE is a class that represents a composition of NeRF_MLP_Residual_Scaled models.

        Args:
            input_dim (int): The input dimension of the model.
            hidden_dim (int): The hidden dimension of the model.
            output_dim (int): The output dimension of the model.
            num_freqs (int or list of length 2, optional): The number of frequencies used for positional encoding. Defaults to 10.
            num_layers (int, optional): The number of layers in each NeRF_MLP_Residual_Scaled model. Defaults to 4.
            num_compose (int, optional): The number of NeRF_MLP_Residual_Scaled models to compose. Defaults to 4.
            normalizing_factor (float, optional): The normalizing factor for layer_id and input_dim. Defaults to 1.0.
        """
        super(NeRF_MLP_Compose_PE, self).__init__()

        # self.positional_encoding = PositionalEncoding(num_freqs, input_dim=input_dim)
        self.output_dim = output_dim
        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        if isinstance(num_freqs, int):
            num_freqs = num_freqs
        elif len(num_freqs) == 2:
            num_freqs = num_freqs[1] - num_freqs[0]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
            
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers))
            
        self.apply(weights_init_uniform_relu)
            
    def forward(self, x, layer_id=None, input_dim=None):
        """
        Forward pass of the NeRF_MLP_Compose_PE model.

        Args:
            x (torch.Tensor): The input tensor.
            layer_id (torch.Tensor, optional): The layer ID tensor. Defaults to None.
            input_dim (torch.Tensor, optional): The input dimension tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        if layer_id is None:
            layer_id = (x[:, 0] * self.norm).int()
        if input_dim is None:
            input_dim = (x[:, -1] * self.norm)
        x[:, :3] = x[:, :3] / x[:, 3:]
        # x = self.positional_encoding(x)
        unique_layer_ids = torch.unique(layer_id)
        output_x = torch.zeros((x.size(0), self.output_dim)).to(x.device)
        for lid in unique_layer_ids:
            mask = lid == layer_id
            output_x[mask] = self.model[lid].forward(x[mask])   
        return output_x / (input_dim.unsqueeze(-1))
    

class NeRF_MLP_Compose_GaussianRFF(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
        """
        NeRF_MLP_Compose_GaussianRFF is a class that represents a composition of NeRF_MLP_Residual_Scaled models.

        Args:
            input_dim (int): The input dimension of the model.
            hidden_dim (int): The hidden dimension of the model.
            output_dim (int): The output dimension of the model.
            num_freqs (int or list of length 2, optional): The number of frequencies used for positional encoding. Defaults to 10.
            num_layers (int, optional): The number of layers in each NeRF_MLP_Residual_Scaled model. Defaults to 4.
            num_compose (int, optional): The number of NeRF_MLP_Residual_Scaled models to compose. Defaults to 4.
            normalizing_factor (float, optional): The normalizing factor for layer_id and input_dim. Defaults to 1.0.
        """
        super(NeRF_MLP_Compose_GaussianRFF, self).__init__()

        self.positional_encoding = GaussianFourierFeatureTransform(num_freqs, input_dim=input_dim)
        self.output_dim = output_dim
        self.model = nn.ModuleList()
        self.norm = normalizing_factor

        if isinstance(num_freqs, int):
            num_freqs = num_freqs
        elif len(num_freqs) == 2:
            num_freqs = num_freqs[1] - num_freqs[0]
        else:
            raise ValueError("num_freqs should be either an integer or a list of length 2.")
            
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers))
            
        self.apply(weights_init_uniform_relu)
            
    def forward(self, x, layer_id=None, input_dim=None):
        """
        Forward pass of the NeRF_MLP_Compose_GaussianRFF model.

        Args:
            x (torch.Tensor): The input tensor.
            layer_id (torch.Tensor, optional): The layer ID tensor. Defaults to None.
            input_dim (torch.Tensor, optional): The input dimension tensor. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        if layer_id is None:
            layer_id = (x[:, 0] * self.norm).int()
        if input_dim is None:
            input_dim = (x[:, -1] * self.norm)
        x[:, :3] = x[:, :3] / x[:, 3:]
        x = self.positional_encoding(x)
        unique_layer_ids = torch.unique(layer_id)
        output_x = torch.zeros((x.size(0), self.output_dim)).to(x.device)
        for lid in unique_layer_ids:
            mask = lid == layer_id
            output_x[mask] = self.model[lid].forward(x[mask])   
        return output_x / (input_dim.unsqueeze(-1))
    
class NeRF_MLP_Compose_MultiResHashEncoding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0):
        """
        NeRF_MLP_Compose_MultiResHashEncoding is a class that represents a composition of NeRF_MLP_Residual_Scaled models.

        Args:
            input_dim (int): The original input dimension to the *hash encoder* (which is now 6 for our 6D model space).
            hidden_dim (int): The hidden dimension of the NeRF_MLP_Residual_Scaled models.
            output_dim (int): The output dimension of the NeRF_MLP_Residual_Scaled models.
            num_freqs (int or list of length 2, optional): The number of frequencies.
            num_layers (int, optional): The number of layers in each NeRF_MLP_Residual_Scaled model. Defaults to 4.
            num_compose (int, optional): The number of NeRF_MLP_Residual_Scaled models to compose. Defaults to 4.
            normalizing_factor (float, optional): The normalizing factor for layer_id and input_dim. Defaults to 1.0.
        """
        super(NeRF_MLP_Compose_MultiResHashEncoding, self).__init__()
        
        # Define the bounding box for the 6D model space (hyperparameters)
        # These ranges need to be consistent with the *normalized* values (v) as per the paper,
        # but for the internal logic of HashEmbedder, it expects the raw values.
        # It's more common to have the normalization *before* feeding to the HashEmbedder.
        # So, the bounding box below should reflect the *expected ranges of the raw inputs*
        # (layer_id, in_channel_id, out_channel_id, total_layers, total_in_channels, total_out_channels)
        # before they are divided by L', N, etc.
        # Let's assume the input to this `NeRF_MLP_Compose_MultiResHashEncoding`
        # is ALREADY the normalized 6D vector `v`.
        # Therefore, the bounding box should be [0.0, 1.0] for all dimensions,
        # since `v` is normalized by L', N, etc.
        
        min_coords_6d = torch.tensor([0.0] * 6, dtype=torch.float32)
        max_coords_6d = torch.tensor([1.0] * 6, dtype=torch.float32) # Normalized values are in [0,1]
        
        self.bounding_box_6d = (min_coords_6d, max_coords_6d)

        # Parameters for HashEmbedder6D
        # The base/finest resolutions apply to the *normalized* 0-1 range.
        # Higher values here mean finer grids within the [0,1] normalized space.
        base_res_6d = [16, 16, 16, 16, 16, 16] # Uniform base resolution for 6 dims
        finest_res_6d = [512, 512, 512, 512, 512, 512] # Uniform finest resolution
        
        # The output dimension of the HashEmbedder6D
        self.hash_encoding_output_dim = 16 * 2 # n_levels * n_features_per_level
        
        self.hash_encoding = HashEmbedder6D(
            bounding_box_6d=self.bounding_box_6d,
            n_levels=16,
            n_features_per_level=2,
            log2_hashmap_size=19,
            base_resolution=base_res_6d,
            finest_resolution=finest_res_6d
        )
        
        self.output_dim = output_dim
        self.num_compose = num_compose
        self.norm = normalizing_factor # This normalizing_factor from the original args might be for layer_id/input_dim extraction.

        if isinstance(num_freqs, int):
            self.effective_num_freqs = num_freqs
        elif isinstance(num_freqs, (list, tuple)) and len(num_freqs) == 2:
            self.effective_num_freqs = num_freqs[1] - num_freqs[0]
        else:
            raise ValueError("num_freqs should be an integer or a list of length 2.")
            
        self.model = nn.ModuleList()
        for _ in range(self.num_compose):
            self.model.append(
                NeRF_MLP_Residual_Scaled(
                    input_dim=self.hash_encoding_output_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_freqs=self.effective_num_freqs,
                    num_layers=num_layers
                )
            )
            
        self.apply(weights_init_uniform_relu)
            
    def forward(self, x_6d_input, layer_id=None, input_dim=None):
        """
        Forward pass of the NeRF_MLP_Compose_MultiResHashEncoding model.

        Args:
            x_6d_input (torch.Tensor): The 6D input tensor representing model space coordinates:
                                       v = [l/L', cin/Cin, cout/Cout, L/N, Cin/N, Cout/N].
                                       Shape: B x 6
            layer_id (torch.Tensor, optional): The layer ID tensor. If None, derived from x_6d_input.
            input_dim (torch.Tensor, optional): The input dimension tensor. If None, derived from x_6d_input.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Ensure `BOX_OFFSETS_6D` is on the correct device for `get_voxel_vertices_6d`
        global BOX_OFFSETS_6D
        if x_6d_input.device != BOX_OFFSETS_6D.device:
            BOX_OFFSETS_6D = BOX_OFFSETS_6D.to(x_6d_input.device)

        # Ensure the hash_encoding and its internal tensors are on the correct device
        # This block ensures all necessary tensors for HashEmbedder6D are on the current device
        if x_6d_input.device != self.hash_encoding.bounding_box_6d[0].device:
            self.hash_encoding.bounding_box_6d = (self.bounding_box_6d[0].to(x_6d_input.device), self.bounding_box_6d[1].to(x_6d_input.device))
            self.hash_encoding.base_resolution = self.hash_encoding.base_resolution.to(x_6d_input.device)
            self.hash_encoding.finest_resolution = self.hash_encoding.finest_resolution.to(x_6d_input.device)
            self.hash_encoding.b = self.hash_encoding.b.to(x_6d_input.device)
            
        # The normalization from the paper happens BEFORE this point.
        # So `x_6d_input` here is `v`.
        
        hash_encoded_x, valid_mask = self.hash_encoding(x_6d_input) # B x (n_levels * n_features_per_level)
        
        # --- Extracting layer_id and `input_dim` from `x_6d_input` ---
        # According to the paper's `v` vector:
        # v[0] = l/L' => layer_id = v[0] * L' (original layer index)
        # v[1] = cin/Cin => input_channel_id = v[1] * Cin (original input channel index)
        # v[2] = cout/Cout => output_channel_id = v[2] * Cout (original output channel index)
        # v[3] = L/N => total_layers = v[3] * N
        # v[4] = Cin/N => total_in_channels = v[4] * N
        # v[5] = Cout/N => total_out_channels = v[5] * N

        # The paper's formulation `x[:, :3] = x[:, :3] / x[:, 3:]` is NOT `v`
        # It's actually `x[:, 0]` being `l`, `x[:, 1]` being `c_in`, etc.
        # And `x[:, 3]` being `L'`, `x[:, 4]` being `C_in`, etc.
        # This means the *input_tensor* to this forward must be `B x 6` where:
        # [l, c_in, c_out, L', C_in, C_out] OR [l, c_in, c_out, L, C_in, C_out] depending on the paper's N constant.
        # Let's re-read the original `forward` logic.
        
        # Original forward logic:
        # if layer_id is None: layer_id = (x[:, 0] * self.norm).int()
        # if input_dim is None: input_dim = (x[:, -1] * self.norm)
        # x[:, :3] = x[:, :3] / x[:, 3:]
        # x = self.hash_encoding(x) <--- PROBLEM HERE if `x` was original non-normalized.

        # The input `x` to `NeRF_MLP_Compose_MultiResHashEncoding.forward` is the raw
        # (layer_id, in_channel_id, out_channel_id, in_layer_size, out_layer_size)
        # from the previous code.
        # But the paper says 6D input `v` is ALREADY NORMALIZED.
        # This implies a normalization step happens *before* this `forward` method.
        # If your `sample_weights` function passes the raw 5D values as `input_tensor`,
        # then the 6th dimension was missing, and the normalization needs to happen here.

        # Let's assume the `x_6d_input` passed to this forward is the `v` vector from the paper,
        # meaning it's already normalized to [0,1].
        # In this case, `layer_id` is the first element of `v`, which is `l/L'`.
        # To get the original `l` (layer_id), we need to multiply by `L'`.
        # However, `L'` is not available directly from `v[0]`. It is `v[3] * N`.
        # This is where the confusion arises. The paper's `v` implies *already normalized*.
        
        # Given your `sample_weights` passes `input_tensor, layer_id=layer_id, input_dim=input_dim`
        # and the original `forward` computes `layer_id = (x[:, 0] * self.norm).int()`,
        # it seems `x[:,0]` is `l/L_prime` (raw layer ID divided by normalizing factor `L_prime`).
        # And `input_dim` is `(x[:, -1] * self.norm)`. The original `x[:, -1]` was `out_layer_size`.
        # The paper's `v` includes `L/N`, `Cin/N`, `Cout/N`.

        # Let's stick to the interpretation that `x_6d_input` is the *normalized* `v` vector.
        # This means `x_6d_input` (the variable, not the mathematical `x` in the paper)
        # IS the `v` vector: `[l/L', cin/Cin, cout/Cout, L/N, Cin/N, Cout/N]`.
        
        # Then, how `layer_id` and `input_dim` were derived from `x` previously:
        # `layer_id = (x[:, 0] * self.norm).int()`
        # `input_dim = (x[:, -1] * self.norm)`
        # This implies that `x[:, 0]` is `l/L'`, and `self.norm` is `L'`.
        # And `x[:, -1]` is `Cout/N`, and `self.norm` here would be `N`. This is inconsistent.

        # The most consistent interpretation given the original `NeRF_MLP_Compose` structure and the paper's `v`:
        # 1. The input `x_6d_input` to this `forward` method *is* the `v` vector.
        # 2. `layer_id` should be derived from `v[0]`. To use it as an index, you need the actual layer ID `l`.
        #    This requires knowing `L'` (the normalizing factor for layer ID).
        #    If `self.norm` in the init of `NeRF_MLP_Compose_MultiResHashEncoding` is `L_prime`, then `l = v[0] * self.norm`.
        # 3. The `input_dim` for the final normalization `output_x / input_dim` in the original code,
        #    if it refers to `in_channel_id` (cin), it's `v[1]`. Or if it's `in_layer_size` from previous.
        #    The paper says `w_ij = MLP(gamma_PE(v)) / C_in`. So `input_dim` should be `C_in`.
        #    `C_in` is not directly `v`'s last element (`Cout/N`). It's derived from `v[4] * N`.

        # This indicates a mismatch between the original `NeRF_MLP_Compose` (which took 5D input and computed `layer_id` and `input_dim` from it)
        # and the paper's 6D `v` vector.

        # **Proposed resolution:**
        # Assume `x_6d_input` is the raw, un-normalized values for:
        # [layer_idx, in_channel_idx, out_channel_idx, total_layers_in_network, total_in_channels_in_network, total_out_channels_in_network]
        # And `self.norm` provides the normalizing constants for `layer_id` and `input_dim_val`.
        # The normalization into `v` happens *here* just before feeding to `hash_encoding`.

        # First, extract raw values for layer_id and input_dim for indexing/normalization
        raw_layer_id = x_6d_input[:, 0]
        # The paper normalizes by Cin. If input_dim from args is the total input channels, use that.
        # Or if it's the `in_layer_size` as in previous contexts, use that.
        # Let's assume `input_dim` refers to the original `in_channel_id` for the denominator.
        # The paper's formula `w_ij = MLP(...) / C_in` means we need `C_in`.
        # If `x_6d_input` has `C_in` at index 4 (i.e., `total_in_channels_in_network`), then that's it.
        
        # Let's align with the `v` vector from the paper:
        # v = [ l/L', c_in/C_in_local, c_out/C_out_local, L/N, C_in_global/N, C_out_global/N ]
        # The `x_6d_input` should contain these 6 normalized values directly.
        # If it DOES, then `layer_id` and `input_dim` need to be derived from these.

        # Given `sample_weights` uses `layer_id=layer_id, input_dim=input_dim` in the call,
        # it seems those are already pre-computed from the original (non-v) inputs.
        # Let's use the provided `layer_id` and `input_dim` if they're given.
        # If not, try to derive them from `x_6d_input` based on the old usage.

        # The line `x[:, :3] = x[:, :3] / x[:, 3:]` implies:
        # x is `[l, c_in, c_out, L', C_in_local, C_out_local]`
        # Then `l` gets divided by `L'`, `c_in` by `C_in_local`, `c_out` by `C_out_local`.
        # This is a normalization step, and it produces the first 3 elements of `v`.
        # The other 3 elements of `v` (`L/N, C_in/N, C_out/N`) would also need to be part of `x`.
        
        # **Crucial Decision Point:**
        # Is `x_6d_input` into this `forward` method:
        # A) The raw un-normalized values like `(layer_id, in_channel_id, ..., normalizing_factors...)`
        # B) The already normalized `v` vector: `[l/L', cin/Cin, cout/Cout, L/N, Cin/N, Cout/N]`
        
        # Based on the error and the paper, it's highly likely the code expects (A) then normalizes,
        # but the `HashEmbedder` was built assuming (B).
        # Let's adjust `NeRF_MLP_Compose_MultiResHashEncoding`'s `forward` to do the normalization.

        # Assuming `x_6d_input` is now 6D, containing:
        # `[raw_layer_id, raw_in_channel_id, raw_out_channel_id, L_prime_for_layer, C_in_for_normalization, N_for_other_dims]`
        # (This is an interpretation to make the original division make sense with 6 dims)
        
        # Original forward had: `x[:, :3] = x[:, :3] / x[:, 3:]`
        # This implies:
        # x_normalized = torch.zeros_like(x_6d_input)
        # x_normalized[:, 0] = x_6d_input[:, 0] / x_6d_input[:, 3] # l / L'
        # x_normalized[:, 1] = x_6d_input[:, 1] / x_6d_input[:, 4] # c_in / C_in
        # x_normalized[:, 2] = x_6d_input[:, 2] / x_6d_input[:, 5] # c_out / C_out
        # The paper's v has 6 elements. If your x_6d_input is like this, then the remaining
        # x_normalized[:, 3], x_normalized[:, 4], x_normalized[:, 5] would be
        # L/N, Cin/N, Cout/N. These would have to be part of `x_6d_input` too.
        # For simplicity and aligning with the paper's `v` definition,
        # let's assume `x_6d_input` already IS the `v` vector, i.e., it's already normalized.
        # This is the most direct interpretation of the paper's notation for `v`.
        # So, we remove the `x[:, :3] = x[:, :3] / x[:, 3:]` line.
        
        # If `layer_id` and `input_dim` are passed as kwargs, use them.
        # Otherwise, derive them from `x_6d_input` based on the paper's `v`.
        
        # 1. `layer_id` for `self.model[lid]` selection:
        # This usually refers to the integer layer index `l`.
        # From `v[0] = l / L'`, we need `l = v[0] * L'`.
        # Since `L'` (the normalizing factor for layer_id) is likely `num_compose-1` or `num_compose`,
        # we can use `self.num_compose` or pass `L_prime` as an arg.
        
        if layer_id is None:
            # Assume `x_6d_input[:, 0]` is `l / L_prime`.
            # If `self.norm` from `__init__` is intended to be `L_prime`, use it.
            # Otherwise, a reasonable default for `L_prime` is `num_compose` or `num_compose - 1`.
            # Let's assume `num_compose` acts as `L_prime`.
            derived_layer_id = (x_6d_input[:, 0] * self.num_compose).int()
            # Clamp derived layer_id to be within valid range [0, num_compose - 1]
            derived_layer_id = torch.clamp(derived_layer_id, 0, self.num_compose - 1)
        else:
            # If `layer_id` is explicitly provided, use it.
            # Ensure it's int and clamped.
            derived_layer_id = layer_id.int()
            derived_layer_id = torch.clamp(derived_layer_id, 0, self.num_compose - 1)


        # 2. `input_dim` for final normalization:
        # Paper says `output / C_in`. `C_in` is `v[4] * N`.
        # But original code used `x[:, -1] * self.norm` where `x[:, -1]` was `out_layer_size`.
        # This is a major semantic conflict.
        # Let's assume `input_dim` is the intended normalizing factor (e.g., `C_in` or `out_layer_size`).
        # If `input_dim` is provided as an argument, use it.
        # If not, and we assume `x_6d_input` is `v`, then we'd need `N` to get `C_in` from `v[4]`.
        # For now, let's keep the original logic for `input_dim` if it's passed as None,
        # but recognize this needs careful alignment with `sample_weights` function.
        if input_dim is None:
            # If `x_6d_input[:, -1]` is `Cout/N`, and `self.norm` is `N`, then this extracts `Cout`.
            # This is a guess based on the existing code structure.
            derived_input_dim = (x_6d_input[:, -1] * self.norm)
        else:
            derived_input_dim = input_dim

        # Now, pass `x_6d_input` directly to the hash encoder.
        hash_encoded_x, valid_mask = self.hash_encoding(x_6d_input) # B x (n_levels * n_features_per_level)
        
        unique_layer_ids = torch.unique(derived_layer_id)
        
        output_x = torch.zeros((hash_encoded_x.size(0), self.output_dim), device=hash_encoded_x.device)
        
        # print('error here hypermodel 703')
        for lid in unique_layer_ids:
            mask = (lid == derived_layer_id)
            output_x[mask] = self.model[lid].forward(hash_encoded_x[mask])
            
        # Ensure division by zero is handled
        input_dim_val_safe = torch.where(derived_input_dim == 0, torch.full_like(derived_input_dim, 1e-6), derived_input_dim)
        return output_x / (input_dim_val_safe.unsqueeze(-1))

class NeRF_ResMLP_Compose_PE(NeRF_MLP_Compose_PE):
    """
    NeRF_ResMLP_Compose is a class that represents a compositional multi-layer perceptron (MLP) model with residual connections.
    
    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The dimensionality of the output.
        num_freqs (int, optional): The number of frequencies used in positional encoding. Defaults to 10.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 4.
        num_compose (int, optional): The number of compositional MLPs to be composed. Defaults to 4.
        normalizing_factor (float, optional): The normalizing factor for the model. Defaults to 1.0.
        scalar (float, optional): The scalar value used in the residual connections. Defaults to 0.1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0, scalar=0.1):
        super(NeRF_ResMLP_Compose_PE, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)
        self.model = nn.ModuleList()
        self.norm = normalizing_factor
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers, scalar=scalar))
            
        self.apply(weights_init_uniform_relu)

class NeRF_ResMLP_Compose_NoEncodings(NeRF_MLP_Compose_NoEncodings):
    """
    NeRF_ResMLP_Compose is a class that represents a compositional multi-layer perceptron (MLP) model with residual connections.
    
    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The dimensionality of the output.
        num_freqs (int, optional): The number of frequencies used in positional encoding. Defaults to 10.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 4.
        num_compose (int, optional): The number of compositional MLPs to be composed. Defaults to 4.
        normalizing_factor (float, optional): The normalizing factor for the model. Defaults to 1.0.
        scalar (float, optional): The scalar value used in the residual connections. Defaults to 0.1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0, scalar=0.1):
        super(NeRF_ResMLP_Compose_NoEncodings, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)
        self.model = nn.ModuleList()
        self.norm = normalizing_factor
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers, scalar=scalar))
            
        self.apply(weights_init_uniform_relu)

class NeRF_ResMLP_Compose_GaussianRFF(NeRF_MLP_Compose_GaussianRFF):
    """
    NeRF_ResMLP_Compose is a class that represents a compositional multi-layer perceptron (MLP) model with residual connections.
    
    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The dimensionality of the output.
        num_freqs (int, optional): The number of frequencies used in positional encoding. Defaults to 10.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 4.
        num_compose (int, optional): The number of compositional MLPs to be composed. Defaults to 4.
        normalizing_factor (float, optional): The normalizing factor for the model. Defaults to 1.0.
        scalar (float, optional): The scalar value used in the residual connections. Defaults to 0.1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0, scalar=0.1):
        super(NeRF_ResMLP_Compose_GaussianRFF, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)
        self.model = nn.ModuleList()
        self.norm = normalizing_factor
        for _ in range(num_compose):
            self.model.append(NeRF_MLP_Residual_Scaled(input_dim + 2 * input_dim * num_freqs, hidden_dim, output_dim, num_freqs, num_layers, scalar=scalar))
            
        self.apply(weights_init_uniform_relu)

class NeRF_ResMLP_Compose_MultiResHashEncoding(NeRF_MLP_Compose_MultiResHashEncoding):
    """
    NeRF_ResMLP_Compose is a class that represents a compositional multi-layer perceptron (MLP) model with residual connections.
    
    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The dimensionality of the hidden layers.
        output_dim (int): The dimensionality of the output.
        num_freqs (int, optional): The number of frequencies used in positional encoding. Defaults to 10.
        num_layers (int, optional): The number of layers in the MLP. Defaults to 4.
        num_compose (int, optional): The number of compositional MLPs to be composed. Defaults to 4.
        normalizing_factor (float, optional): The normalizing factor for the model. Defaults to 1.0.
        scalar (float, optional): The scalar value used in the residual connections. Defaults to 0.1.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_freqs=10, num_layers=4, num_compose=4, normalizing_factor=1.0, scalar=0.1):
        super(NeRF_ResMLP_Compose_MultiResHashEncoding, self).__init__(input_dim, hidden_dim, output_dim, num_freqs, num_layers, num_compose, normalizing_factor)
        self.model = nn.ModuleList()
        self.norm = normalizing_factor
        for _ in range(num_compose):
            mlp_input_dim = self.hash_encoding_output_dim # Get the output dim from the hash encoder
            self.model.append(
                NeRF_MLP_Residual_Scaled(
                    input_dim=mlp_input_dim, # This will be 32
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_freqs=None, # num_freqs is no longer relevant for input_dim, explicitly set to None
                    num_layers=num_layers,
                    scalar=scalar
                )
            )            
        self.apply(weights_init_uniform_relu)