import torch
import torchvision

def quantize(T, stair):
  batches = torch.round((T*255-0.000001)/stair)
  new_color = (batches*stair)/(255)
  return new_color

class Downsampling(torch.nn.Module):
  """
  Changes resolution of a given tensor
  """
  def __init__(self, downsampling_percentage, imagenet=True):
    super().__init__()
    if downsampling_percentage<0: downsampling_percentage = 0.02
    if downsampling_percentage>1: downsampling_percentage = 1
    self.downsampling_percentage = downsampling_percentage
    if imagenet: self.max_res = 224

  def forward(self, img):
    B, C, H, W = img.shape
    img = torchvision.transforms.Resize((int(H*self.downsampling_percentage), int(W*self.downsampling_percentage)))(img)
    img = torchvision.transforms.Resize(self.max_res)(img)
    return img

class Quantization(torch.nn.Module):
  """
  Quantization
  """
  def __init__(self, actual, verbose=True):
    super().__init__()
    self.actual = actual*255.0
    self.bits = int(round(self.actual))
    if self.bits>255: self.bits = 1
    if self.bits<1: self.bits = 1
    self.stair = int(round(255/self.bits))
    self.boxes = [float(i)/255 for i in range(0,256, self.stair)]
    self.len_boxes = len(self.boxes)
    if verbose: print(f'Cantidad de colores por canal: {self.len_boxes}')


  def forward(self, img):
    return quantize(img, self.stair)

class CroppingTop(torch.nn.Module):
  """
  Cropps an image from top to bottom
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(H*self.crop_percentage)
    l = list(range(crop))
    if len(l) == 0: return img
    h = torch.tensor(l).to(self.device)
    mask.index_fill_(0, h, 0)
    return img*mask

class CroppingBottom(torch.nn.Module):
  """
  Cropps an image from bottom to top
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(H-H*self.crop_percentage)
    l = list(range(crop,H))
    if len(l) == 0: return img
    h = torch.tensor(l).to(self.device)
    mask.index_fill_(0, h, 0)
    return img*mask

class CroppingLeft(torch.nn.Module):
  """
  Cropps an image from left to right
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = device

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(W*self.crop_percentage)
    l = list(range(crop))
    if len(l) == 0: return img
    v = torch.tensor(l).to(self.device)
    mask.index_fill_(1, v, 0)
    return img*mask

class CroppingRight(torch.nn.Module):
  """
  Cropps an image from right to left
  """
  def __init__(self, crop_percentage, device='cuda'):
    super().__init__()
    if crop_percentage<0: crop_percentage = 0
    if crop_percentage>1: crop_percentage = 1
    self.crop_percentage = crop_percentage
    self.device = 'cuda'

  def forward(self, img):
    _, C, H, W = img.shape
    mask = torch.ones((H,W)).to(self.device)
    crop = int(W-W*self.crop_percentage)
    l = list(range(crop,W))
    if len(l) == 0: return img
    v = torch.tensor(l).to(self.device)
    mask.index_fill_(1, v, 0)
    return img*mask
