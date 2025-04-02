import os
import torch
from torch import nn
from torch.nn import functional as F
import errno
import nibabel as nib
from torchvision import utils
import torchvision.models as models
import numpy as np


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def reparameterize_logit(logit):
    import warnings
    warnings.filterwarnings('ignore', message='.*Mixed memory format inputs detected.*')
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta


def save_image(images, file_name):
    image_save = torch.cat([image[:4, [0], ...].cpu() for image in images], dim=0)
    image_save = utils.make_grid(tensor=image_save, nrow=4, normalize=False, range=(0, 1)).detach().numpy()[0, ...]
    image_save = nib.Nifti1Image(image_save.transpose(1, 0), np.eye(4))
    nib.save(image_save, file_name)


def dropout_contrasts(available_contrast_id, contrast_id_to_drop=None):
    """
    Randomly dropout contrasts for HACA3 training.

    ==INPUTS==j
    * available_contrast_id: torch.Tensor (batch_size, num_contrasts)
        Indicates the availability of each MR contrast. 1: if available, 0: if unavailable.

    * contrast_id_to_drop: torch.Tensor (batch_size, num_contrasts)
        If provided, indicates the contrast indexes forced to drop. Default: None

    ==OUTPUTS==
    * contrast_id_after_dropout: torch.Tensor (batch_size, num_contrasts)
        Some available contrasts will be randomly dropped out (as if they are unavailable).
        However, each sample will have at least one contrast available.
    """
    batch_size = available_contrast_id.shape[0]
    if contrast_id_to_drop is not None:
        available_contrast_id = available_contrast_id - contrast_id_to_drop
    contrast_id_after_dropout = available_contrast_id.clone()
    for i in range(batch_size):
        available_contrast_ids_per_subject = (available_contrast_id[i] == 1).nonzero(as_tuple=False).squeeze(1)
        num_available_contrasts = available_contrast_ids_per_subject.numel()
        if num_available_contrasts > 1:
            num_contrast_to_drop = torch.randperm(num_available_contrasts - 1)[0]
            contrast_ids_to_drop = torch.randperm(num_available_contrasts)[:num_contrast_to_drop]
            contrast_ids_to_drop = available_contrast_ids_per_subject[contrast_ids_to_drop]
            contrast_id_after_dropout[i, contrast_ids_to_drop] = 0.0
    return contrast_id_after_dropout


class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model):
        super().__init__()
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.children())[:13]).eval()

    def forward(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        return F.l1_loss(self.vgg(x), self.vgg(y))


class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature

    def forward(self, query_feature, positive_feature, negative_feature):
        B, C, N = query_feature.shape

        l_positive = (query_feature * positive_feature).sum(dim=1)[:, :, None]
        l_negative = torch.bmm(query_feature.permute(0, 2, 1), negative_feature)

        logits = torch.cat((l_positive, l_negative), dim=2) / self.temperature

        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * N, dtype=torch.long).to(query_feature.device)
        return self.ce_loss(predictions, targets).mean()


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        kld_loss = -0.5 * logvar + 0.5 * (torch.exp(logvar) + torch.pow(mu, 2)) - 0.5
        return kld_loss


def divide_into_batches(in_tensor, num_batches):
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image, image_dim=256):
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded

def zero_pad2d(image, image_dim=256):
    [n_row, n_col] = image.shape
    image_padded = np.zeros((image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2] = image
    return image_padded


def crop(image, n_row, n_col, n_slc):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]

def crop2d(image, n_row, n_col):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2]

def normalize_attention(attention_map):
    """
    Normalize the attention map such that the channels sum to 1.
    - If all channels are 0, distribute attention equally.
    - If some channels are 0, normalize the remaining non-zero values to sum to 1.

    Args:
    - attention_map (torch.Tensor): Attention map of shape [batch_size, num_contrasts, height, width].
    - attention_map (torch.Tensor): Attention map of shape [batch_size, height, width, num_contrasts].
    
    Returns:
    - torch.Tensor: Normalized attention map.
    """

    # print(f"attention_map type: {attention_map.dtype}, shape: {attention_map.shape}")

    # Sum over the channels dimension (dim=3)
    attention_sum = attention_map.sum(dim=3, keepdim=True)  # Shape: [batch_size, height, width, 1]
    # print(f"attention_sum type: {attention_sum.dtype}, shape: {attention_sum.shape}")

    # Find where all channels are ~0
    zero_sum_mask = (attention_sum < 1e-6)  # Shape: [batch_size, height, width, 1]
    # print(f"zero_sum_mask type: {zero_sum_mask.dtype}, shape: {zero_sum_mask.shape}")

    # Set all-zero areas to equal weighting across channels
    num_contrasts = attention_map.size(3)
    # print(f"num_contrasts: {num_contrasts}")
    
    attention_map[zero_sum_mask.expand_as(attention_map)] = 1.0 / num_contrasts

    # Recalculate the sum after handling all-zero channels
    attention_sum = attention_map.sum(dim=3, keepdim=True)  # Recompute after handling zero cases
    # print(f"attention_sum type: {attention_sum.dtype}, shape: {attention_sum.shape}")

    # Normalize non-zero attention values
    attention_map = attention_map / (attention_sum + 1e-6)  # Avoid division by zero
    # print(f"attention_map type: {attention_map.dtype}, shape: {attention_map.shape}")

    return attention_map

def gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    """Create a 2D Gaussian kernel."""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def smooth_attention(attention_map, kernel_size=5, sigma=1.0):
    """
    Smooth the attention map using a 2D Gaussian filter.
    
    Args:
        attention_map (torch.Tensor): Shape [B, H, W] or [B, H, W, C]
        kernel_size (int): Size of the Gaussian kernel
        sigma (float): Standard deviation of the Gaussian

    Returns:
        torch.Tensor: Smoothed attention map of the same shape
    """
    original_shape = attention_map.shape
    device = attention_map.device

    # Handle [B, H, W, C] -> [B*C, 1, H, W] format
    if attention_map.dim() == 4:
        B, H, W, C = attention_map.shape
        x = attention_map.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    elif attention_map.dim() == 3:
        B, H, W = attention_map.shape
        x = attention_map.unsqueeze(1)  # [B, 1, H, W]
    else:
        raise ValueError("attention_map must be 3D or 4D")

    kernel = gaussian_kernel(kernel_size, sigma, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    kernel = kernel.to(dtype=x.dtype)

    padding = kernel_size // 2
    x_smooth = F.conv2d(x, kernel, padding=padding, groups=1)

    # Reshape back to original
    if attention_map.dim() == 4:
        x_smooth = x_smooth.view(B, C, H, W).permute(0, 2, 3, 1)
    else:
        x_smooth = x_smooth.squeeze(1)

    return x_smooth
