## Common functions and imports
import numpy as np
from PIL import Image
import skimage
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import ipdb

interpolation = 'bilinear'
# interpolation = 'nearest'
bias = True
include_lowres = False

resolutions1 = [32, 64, 128, 192, 256, 320, 384, 448, 512]
# resolutions2 = [16, 32, 64, 128, 256]
resolutions2 = [1, 8, 16, 32, 64]
feature_dims = [32, 64, 128]
hidden_dims = [32, 64, 128, 256]
sparse_dims = []
topks = []


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


img = skimage.data.astronaut() / 255
# Convert to grayscale for simplicity
if len(img.shape) > 2:
  img = np.mean(img, axis=-1)


# make a low resolution version of the image
def lowres(img, shrink_factor=None, model_size=None, resample=Image.BILINEAR):
  im = Image.fromarray(np.array(img*255, dtype=np.uint8))
  width, height = im.size
  if shrink_factor is None:  # autocompute the shrink factor to achieve a certain model size
    assert model_size is not None
    shrink_factor = int(np.sqrt(height * width / model_size))
  newsize = (width//shrink_factor, height//shrink_factor)
  if newsize[0] < 1 or newsize[1] < 1:
    return np.zeros(img.shape)
  im_small = im.resize(newsize, resample=resample)
  return np.array(im_small.resize((width, height), resample=resample)) / 255, newsize[0] * newsize[1]


# make a low rank approximation of the image
def lowrank(residual, rank_reduction=None, model_size=None):
  assert residual.shape[0] == residual.shape[1]
  if rank_reduction is None:  # autocompute the rank reduction to achieve a certain model size
    assert model_size is not None
    rank_reduction = 2*(residual.shape[0]**2) // model_size
  keep_rank = residual.shape[0]//rank_reduction
  if keep_rank < 1:
    return np.zeros_like(residual), 0
  U, S, Vh = np.linalg.svd(residual, full_matrices=True)
  S[keep_rank:] = 0
  smat = np.diag(S)
  low_rank = np.real(U @ smat @ Vh)
  return low_rank, keep_rank * residual.shape[0] * 2


# make a sparse approximation of the image
def sparse(residual, sparse_reduction=None, model_size=None):
  flat = residual.flatten()
  if sparse_reduction is None:  # autocompute s to achieve a certain model size
    s = model_size // 2
  else:
    s = len(flat) // (sparse_reduction**2)
  result = np.zeros_like(flat)
  idx = np.flip(np.argsort(residual.flatten()))[0:s] # no need to take absolute value since image values are all nonnegative
  result[idx] = flat[idx]
  return result.reshape(residual.shape), s * 2

def psnr_normalized(gt, recon):
  # First compute the normalization factor based on the gt image
  scale = 1
  if np.max(gt) > 1:
    scale = 255
  residual = (gt - recon) / scale
  mse = np.mean(np.square(residual))
  return -10*np.log10(mse)


# Filter a set of points to find those that are pareto optimal
# assuming higher is better for y and smaller is better for x
def find_pareto(xvals, yvals):
  # First sort according to xvals, in increasing order
  idx = np.argsort(xvals)
  xvals = xvals[idx]
  yvals = yvals[idx]
  # Calculate pareto frontier indices
  pareto_indices = []
  for i in range(len(xvals)):
    is_pareto = True
    for j in range(len(xvals)): # was 0.015
      if i != j and xvals[j] <= xvals[i] + 0.0 and yvals[j] >= yvals[i]:  # TODO: make this a little more restrictive
        is_pareto = False
        break
    if is_pareto:
      pareto_indices.append(i)
  return xvals[pareto_indices], yvals[pareto_indices], idx[pareto_indices]




class CustomModel(nn.Module):
    def __init__(self, dim1, dim2, dim_features, m, resolution, operation, decoder, interpolation, bias, mode, sdim=None):
        super(CustomModel, self).__init__()

        self.resx, self.resy = resolution
        self.operation = operation
        self.decoder = decoder
        self.interpolation = interpolation
        self.bias = bias
        self.mode = mode # lowres, sparse, sparse_lowres

        
        # Define the feature tensors
        torch.manual_seed(0)
        if operation == 'multiply':
          self.line_feature_x = nn.Parameter(torch.rand(dim_features, dim1)*0.15 + 0.1)  # can set different line and plane feature dims
          self.line_feature_y = nn.Parameter(torch.rand(dim_features, dim1)*0.15 + 0.1)
        else:
          self.line_feature_x = nn.Parameter(torch.rand(dim_features, dim1)*0.03 + 0.005) 
          self.line_feature_y = nn.Parameter(torch.rand(dim_features, dim1)*0.03 + 0.005)
        # print(self.line_feature_x[:,0])

        if "lowres" in (self.mode):
            self.plane_feature = nn.Parameter(torch.randn(dim_features, dim2, dim2)*0.01)

        if "sparse" in (self.mode):
            self.plane_feature_sparse = nn.Parameter(torch.randn(dim_features, sdim, sdim)*0.01)
        
        # Define the decoder
        if decoder == 'linear':
          self.mlp = nn.Sequential(
            nn.Linear(dim_features, 1, bias=self.bias),
          )
        elif decoder == 'nonconvex':
          self.mlp = nn.Sequential(
              nn.Linear(dim_features, m, bias=self.bias),
              nn.ReLU(),
              nn.Linear(m, 1, bias=self.bias)
          )
        elif decoder == 'convex':
          self.fc1 = nn.Linear(dim_features, m, bias=self.bias)
          self.fc2 = nn.Linear(dim_features, m, bias=self.bias)
          self.fc2.weight.requires_grad = False
          if self.bias:
            # stdv = 1. / np.sqrt(self.fc2.weight.size(1))
            # self.fc2.bias.data.uniform_(0, stdv/10)  # so far no variations on this are helpful
            self.fc2.bias.requires_grad = False

        else:
          raise ValueError(f"Invalid decoder {decoder}; expected linear, nonconvex, or convex")


    def forward(self, coords):
        # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)  # [batchx, batchy, 1]
        y_coords = coords[..., 1].unsqueeze(-1)  # [batchx, batchy, 1]

        # Scale to [-1, 1] range for grid_sample
        x_coords = (x_coords * 2 / self.resx) - 1
        y_coords = (y_coords * 2 / self.resy) - 1
        
        # Combine x and y coordinates
        gridx = torch.cat((x_coords, x_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        gridy = torch.cat((y_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]

        # Interpolate line features using grid_sample
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)  # [1, dim_features, dim1, 1]
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)  # [1, dim_features, dim1, 1]

        # Get the feature tensors for grid_sample
        feature_x = F.grid_sample(line_features_x, gridx, mode=self.interpolation, padding_mode='border', align_corners=True)  # [1, dim_features, batchx, batchy]
        feature_y = F.grid_sample(line_features_y, gridy, mode=self.interpolation, padding_mode='border', align_corners=True)  # [1, dim_features, batchx, batchy]

        # Prepare for 2D interpolation for the plane feature
        if "lowres" in (self.mode):
            plane_features = self.plane_feature.unsqueeze(0)  # [1, dim_features, dim2, dim2]
        if "sparse" in (self.mode):
            plane_features_sparse = self.plane_feature_sparse.unsqueeze(0)  # [1, dim_features, dim2, dim2]
        plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]

        # Sample from the plane feature using grid_sample
        if self.mode == "lowres":
            sampled_plane_features = F.grid_sample(plane_features, plane_grid, mode=self.interpolation, align_corners=True)
        elif self.mode == "sparse":
            sampled_plane_features = F.grid_sample(plane_features_sparse, plane_grid, mode=self.interpolation, align_corners=True)  # [1, dim_features, batchx, batchy]
        elif self.mode == "sparse_lowres":
            sampled_plane_features = (F.grid_sample(plane_features, plane_grid, mode=self.interpolation, align_corners=True)
                                    + F.grid_sample(plane_features_sparse, plane_grid, mode=self.interpolation, align_corners=True)) # could be concat instead of add

        # Combine features
        if self.operation == 'add':
            combined_features = feature_x + feature_y + sampled_plane_features  # [1, dim_features, batchx, batchy]
        elif self.operation == 'multiply':
            combined_features = feature_x * feature_y + sampled_plane_features  # [1, dim_features, batchx, batchy]
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or multiply")

        # Reorder axes so this can be fed to the MLP
        combined_features = combined_features.squeeze().permute(1, 2, 0)  # [batchx, batchy, dim_features]

        # Pass through decoder
        if self.decoder == 'linear' or self.decoder == 'nonconvex':
          output = self.mlp(combined_features).squeeze()  # [batchx, batchy]
        else:  # convex
          output = self.fc1(combined_features) * (self.fc2(combined_features) > 0)  # [batchx, batchy, m]
          output = torch.mean(output, dim=-1)  # [batchx, batchy]
        
        return output

def topk_filter(tensor, k):

    # Find the kth largest magnitude using torch.topk
    top_k_values, _ = torch.topk(tensor.flatten().abs(), k)
    threshold = top_k_values[-1]  # The smallest of the top-k values
    mask = tensor.abs() >= threshold
    filtered_tensor = tensor * mask
    return filtered_tensor

def sparse_project(model, k):
    # k is the number of elements kept 
    with torch.no_grad():
        model.plane_feature_sparse.data = topk_filter(model.plane_feature_sparse, k)

def model_size(model):
  # exclude zeros from param count
  return sum(torch.count_nonzero(p) for p in model.parameters() if p.requires_grad)

def model_size_quantized(model, bits=4):
  if hasattr(model, 'base_model'):
    model = model.base_model
  
  params = sum(torch.numel(p) for p in model.mlp.parameters() if p.requires_grad) # decoder params
  params += torch.numel(model.line_feature_x)*2 # line features
  if hasattr(model, "plane_feature_sparse"):
    params += torch.count_nonzero(model.plane_feature_sparse) * 2
  if hasattr(model, "plane_feature"):
    # quantized
    params = params.float()
    params += torch.numel(model.plane_feature) * bits / 32
  return params

def experiment(lineres, planeres, dim_features, m, operation, decoder, interpolation, bias, mode, sdim=None, sparse_percent=None):

    # Hyperparameters
    num_epochs = 1000 
    if operation == 'multiply':
      if decoder == 'nonconvex':
        learning_rate = 0.05
      else:
        learning_rate = 0.15
    else:
      if decoder == 'nonconvex':
        learning_rate = 0.005
      else:
        learning_rate = 0.05

    # Get the ground truth image
    img = skimage.data.astronaut() / 255
    # Convert to grayscale for simplicity
    if len(img.shape) > 2:
      img = np.mean(img, axis=-1)

    # Get the dimensions of the image
    xdim, ydim = img.shape
    # print(xdim*ydim)

    # Generate pixel coordinates
    y_indices, x_indices = np.indices((xdim, ydim))
    coords = np.stack((x_indices, y_indices), axis=-1)  # [xdim, ydim, 2]

    # Convert to PyTorch tensors
    coords = torch.from_numpy(coords).float().to(device)
    targets = torch.from_numpy(img).float().to(device)


    # Instantiate the model
    model = CustomModel(dim1=lineres, dim2=planeres, dim_features=dim_features, m=m, resolution=img.shape, operation=operation, decoder=decoder, interpolation=interpolation, bias=bias, mode=mode, sdim=sdim).to(device)
    
    # sparse top-k constant k
    if "sparse" in mode:
        k = int((sparse_percent / 100) * torch.numel(model.plane_feature_sparse))
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop (full batch)
    for epoch in range(num_epochs):
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(coords)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        #### sparse projection here
        if "sparse" in mode:
            sparse_project(model, k)

        # # Print the loss for this epoch
        # if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], PSNR: {-10*np.log10(loss.item()):.4f}')

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(coords)
        # final_outputs = torch.clamp(final_outputs, min=0, max=1)
        plt.figure()
        plt.imshow(final_outputs.squeeze().cpu(), cmap='gray')
        plt.title('prediction')
        plt.colorbar()
        plt.savefig('prediction.jpg')
        plt.close()
        # plt.figure()
        # plt.imshow(targets, cmap='gray')
        # plt.title('ground truth')
        # plt.colorbar()
        final_loss = criterion(final_outputs.squeeze(), targets)
        psnr = -10*np.log10(final_loss.item())
        print(f'Final PSNR: {psnr:.4f}')
        size = model_size(model)
        print(f"Num model parameters: {size}")

    return psnr, size

class QuantizedModel(nn.Module):
    def __init__(self, base_model, bits=4):
        super().__init__()
        self.base_model = base_model
        self.bits = bits

    def forward(self, x):
        if self.training:  # Apply quantization only during training
            with torch.no_grad():
                for param in self.base_model.plane_feature:
                    # ipdb.set_trace()
                    param.data = QuantizationSTE.apply(param.data, self.bits)

        return self.base_model(x)

class QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bits):
        """ Quantizes x to m-bit fixed-point representation """
        qmin = 0
        qmax = 2 ** bits - 1  # Number of levels

        x_norm = (x - x.min()) / (x.max() - x.min()) # normalize to [0,1]
        x_scaled = x_norm * qmax  # Scale to integer range
        x_rounded = torch.round(x_scaled)  # Quantize
        x_quantized = x_rounded / qmax  # Scale back ([0,1]) ## would it cause problems?
        # x_quantized = x_quantized * (x.max()-x.min()) + x.min()  # Scale back (min, max) ??

        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        """ STE: Pass gradients as if no quantization happened """
        return grad_output, None  # None for bits since it's not learnable

def experiment_quantized(img, lineres, planeres, dim_features, m, operation, decoder, interpolation, bias, mode, sdim=None, sparse_percent=None):

    # Hyperparameters
    num_epochs = 1000 
    if operation == 'multiply':
      if decoder == 'nonconvex':
        learning_rate = 0.05
      else:
        learning_rate = 0.15
    else:
      if decoder == 'nonconvex':
        learning_rate = 0.005
      else:
        learning_rate = 0.05

    # # Get the ground truth image
    # img = skimage.data.astronaut() / 255
    # # Convert to grayscale for simplicity
    # if len(img.shape) > 2:
    #   img = np.mean(img, axis=-1)

    ### img assumes range 0-1

    # Get the dimensions of the image
    xdim, ydim = img.shape
    # print(xdim*ydim)

    # Generate pixel coordinates
    y_indices, x_indices = np.indices((xdim, ydim))
    coords = np.stack((x_indices, y_indices), axis=-1)  # [xdim, ydim, 2]

    # Convert to PyTorch tensors
    coords = torch.from_numpy(coords).float().to(device)
    targets = torch.from_numpy(img).float().to(device)


    # Instantiate the model
    # mode should be lowres or lowres_sparse
    model = CustomModel(dim1=lineres, dim2=planeres, dim_features=dim_features, m=m, resolution=img.shape, operation=operation, decoder=decoder, interpolation=interpolation, bias=bias, mode=mode, sdim=sdim).to(device)
    quantized_model = QuantizedModel(base_model=model, bits=4).to(device)
    # sparse top-k constant k
    if "sparse" in mode:
        k = int((sparse_percent / 100) * torch.numel(model.plane_feature_sparse))
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(quantized_model.parameters(), lr=learning_rate)

    # Training loop (full batch)
    for epoch in tqdm(range(num_epochs)):
        quantized_model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = quantized_model(coords)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        #### sparse projection here
        if "sparse" in mode:
            sparse_project(quantized_model.base_model, k)

        # # Print the loss for this epoch
        # if (epoch + 1) % 10 == 0:  # Print every 10 epochs
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], PSNR: {-10*np.log10(loss.item()):.4f}')

    # Final evaluation
    quantized_model.eval()
    with torch.no_grad():
        final_outputs = quantized_model(coords)
        # final_outputs = torch.clamp(final_outputs, min=0, max=1)
        plt.figure()
        plt.imshow(final_outputs.squeeze().cpu(), cmap='gray')
        plt.title('prediction')
        plt.colorbar()
        plt.savefig('prediction.jpg')
        plt.close()
        # plt.figure()
        # plt.imshow(targets, cmap='gray')
        # plt.title('ground truth')
        # plt.colorbar()
        final_loss = criterion(final_outputs.squeeze(), targets)
        psnr = -10*np.log10(final_loss.item())
        print(f'Final PSNR: {psnr:.4f}')
        size = model_size_quantized(model)
        print(f"Num model parameters: {size}")

    return psnr, size
