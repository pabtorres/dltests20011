import itertools
from entropy_reductions import *
from torchvision import transforms

class TNode():
  def __init__(self, pond, crop_jump, res_jump, quant_jump):
      self.crop_jump = crop_jump
      self.res_jump = res_jump
      self.quant_jump = quant_jump
      self.pond = pond # ('2', '2', '3', '4')
      self.cl = 0 # cr = crop left
      self.cr = 0 # cr = crop right
      self.ct = 0 # ct = crop top
      self.cb = 0 # cb = crop bottom
      self.res = 0 # res = resolution
      self.quant = 0 # quant = quantization
      self.transform = None

  def reinit_pond_vars(self):
      self.cl = 0 # cr = crop left
      self.cr = 0 # cr = crop right
      self.ct = 0 # ct = crop top
      self.cb = 0 # cb = crop bottom
      self.res = 0
      self.quant = 0

  def make_transformation(self):
    self.reinit_pond_vars()
    for el in self.pond:
      if el == '1': self.cl+=1
      if el == '2': self.cr+=1
      if el == '3': self.ct+=1
      if el == '4': self.cb+=1
      if el == '5': self.res+=1
      if el == '6': self.quant+=1
    self.transform = eval(f'transforms.Compose([Downsampling({1-self.res*self.res_jump}), Quantization({1-self.quant*self.quant_jump}, verbose={False}), CroppingTop({self.ct*self.crop_jump}), CroppingLeft({self.cl*self.crop_jump}), CroppingBottom({self.cb*self.crop_jump}), CroppingRight({self.cr*self.crop_jump}),])')
    return self.transform
    
  def get_transform(self):
    return self.transform

  def get_pond(self):
    return self.pond
  
  def get_pond_vars(self):
    return {'res': self.res, 'quant': self.quant, 'crop left': self.cl, 'crop right': self.cr, 'crop top': self.ct, 'crop bottom': self.cb}

def generate_graph(crop_jump, res_jump, quant_jump, alpha):
  transformations = []
  for floor in range(1, alpha+1):
    nodes = [TNode(pond, crop_jump, res_jump, quant_jump) for pond in list(itertools.combinations_with_replacement('123456',floor))]
    transformations += [node.make_transformation() for node in nodes]
  return transformations