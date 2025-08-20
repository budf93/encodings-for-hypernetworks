import torch
import torch.nn as nn
from neumeta.models import create_model_cifar10
from neumeta.utils import sample_merge_model, validate_single, get_cifar10, parse_args, load_checkpoint, get_hypernet  # Added get_hypernet
from omegaconf import OmegaConf

# Parse arguments
args = parse_args()
args = OmegaConf.merge(OmegaConf.load(args.config), args)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load validation data
_, val_loader = get_cifar10(args.training.batch_size)

val_criterion = nn.CrossEntropyLoss().to(device)  # Move criterion to GPU

# Calculate number_param from base model
base_model = create_model_cifar10(args.model.type, hidden_dim=64, path=args.model.pretrained_path, smooth=args.model.smooth).to(device)
number_param = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
print(f"Number of parameters in base model: {number_param}")

# Load the trained INR model (hyper_model)
hyper_model_checkpoint = "toy/experiments/resnet20_cifar10_32-64-4layer-200e-noisecoord-resmlpv2_smooth_5_256_16_multires_hash_encoding/model.pth"
hyper_model = get_hypernet(args, number_param).to(device)  # Move to GPU immediately
state_dict = torch.load(hyper_model_checkpoint, map_location=device)
hyper_model.load_state_dict(state_dict)
hyper_model.eval()

# Sample weights for different hidden dimensions
for hidden_dim in range(16, 65):
    model = create_model_cifar10(
        args.model.type,
        hidden_dim=hidden_dim,
        path=args.model.pretrained_path,
        smooth=args.model.smooth
    ).to(device)

    accumulated_model = sample_merge_model(hyper_model, model, args, K=100)
    val_loss, acc = validate_single(accumulated_model, val_loader, val_criterion, args=args)
    print(f"Test using model {args.model.type}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")