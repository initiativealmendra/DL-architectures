import torch
import torch.nn as nn
import torch.nn.functional as F

class attentiveConv(nn.Module):
  """
     Perfoms multihead attention. 
     Args: 
      in_channels : (int) Number of input filters.
      out_channls : (int) Number of output filters. 
      k_size: (int) Kernel size.
      dk: (int) Depth of queries and keys in multihead attention. (usually 2*out_channels)
      dv: (int) Depth of values. (usually out_channels * a value between 0 and 1)
      nh: (int) Number of heads. (usual values are 4, 6, and 8)
      relative: (bool) True for relative positional encodings.
     Return: 
      (tensor) Attention augmented convolution.
  """
  def __init__(self, in_channels, out_channels, k_size, dk, dv, nh, relative=False):
    super(attentiveConv, self).__init__()
    self.f_in = in_channels
    self.f_out = out_channels
    self.k = k_size
    self.dk = dk 
    self.dv = dv
    self.nh = nh
    self.relative = relative
    
    assert self.nh != 0, "nh must be equal or greater than 1"
    assert self.dk % self.nh == 0, "dk must be divisible by nh"
    assert self.dv % self.nh == 0, "dv must be divisible by nh"
    
    self.conv_out = nn.Conv2d(self.f_in, out_channels=self.f_out-self.dv, kernel_size=self.k, padding=(self.k-1)//2)
    self.qkv_conv = nn.Conv2d(self.f_in, out_channels= 2*self.dk+self.dv, kernel_size=self.k, padding=(self.k-1)//2)
    self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1)

  def forward(self, x):
    conv_out = self.conv_out(x)
    B, _, H, W = conv_out.size()
    
    flat_q, flat_k, flat_v, q = self.compute_flat_qkv(x, self.dk, self.dv, self.nh)
    # attention = softmax((keys * queriesT) + rel_pos_w + rel_pos_h) * values
    logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
    if self.relative:
      h_rel_logits, w_rel_logits = self.relative_logits(q)
      logits += h_rel_logits
      logits += w_rel_logits 
    weights = nn.functional.softmax(logits, dim=-1)
    attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
    # Reshaped into a tensor (H,W, dv) to match the original spatial dimensions.
    attn_out = torch.reshape(attn_out, (B, self.nh, self.dv // self.nh, H, W))
    attn_out = self.combine_heads_2d(attn_out)
    attn_out = self.attn_out(attn_out)

    return torch.cat((conv_out, attn_out), dim=1)
  
  def split_heads_2d(self, x, nh):
    """
      Splits channels into multiple heads.
    """
    B, C, H, W = x.size()
    ret_shape = (B, nh, C // nh, H, W)
    return torch.reshape(x, ret_shape)
  
  def combine_heads_2d(self, x):
    """
      Combine miltiple heads.
    """
    B, nh, dv, H, W = x.size()
    ret_shape = (B, nh*dv, H, W)
    return torch.reshape(x, ret_shape)
  
  def compute_flat_qkv(self, x, dk, dv, nh):
    """
      Computes flattenend queries, keys, and values. 
    """
    qkv = self.qkv_conv(x)
    B, _, H, W = qkv.size()
    q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
    q = self.split_heads_2d(q, nh)
    k = self.split_heads_2d(k, nh)
    v = self.split_heads_2d(v, nh)
    dkh = dk // nh
    q *= dkh**-0.5
    flat_q = torch.reshape(q, (B, nh, dk // nh, H*W))
    flat_k = torch.reshape(k, (B, nh, dk // nh, H*W))
    flat_v = torch.reshape(v, (B, nh, dv // nh, H*W))
    return flat_q, flat_k, flat_v, q
  
  def rel_to_abs(self, x):
    """
      Convert tensor from relative to absolute indexing.
    """
    B, nh, L, _ = x.size()
    col_pad = torch.zeros((B, nh, L, 1)).to(x)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = torch.reshape(x, (B, nh, L*2*L))
    flat_pad = torch.zeros((B, nh, L-1)).to(x) 
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = torch.reshape(flat_x_padded, (B, nh, L+1, 2*L-1))
    return final_x[:, :, :L, L-1:]
  
  def relative_logits_1d(self, q, rel_k, H, W, nh, case):
    """
      Computes relative logits alonge one dimension.
    """
    rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
    rel_logits = torch.reshape(rel_logits, (-1, nh*H, W, 2*W-1))
    rel_logits = self.rel_to_abs(rel_logits)
    
    rel_logits = torch.reshape(rel_logits, (-1, nh, H, W, W))
    rel_logits = torch.unsqueeze(rel_logits, dim=3)
    rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))
    
    if case == "w":
      rel_logits = torch.transpose(rel_logits, 3, 4)
    elif case == "h":
      rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
    return torch.reshape(rel_logits, (-1, nh, H*W, H*W))
  
  def relative_logits(self, q):
    """
      Computes relative positions logits. 
    """
    B, nh, dk, H, W = q.size()
    q = torch.transpose(q, 2, 4).transpose(2, 3)
    key_rel_w = nn.Parameter(torch.randn((2*W-1, dk), requires_grad=True)).to(device)
    rel_logits_w = self.relative_logits_1d(q, key_rel_w, H, W, nh, "w")
    
    key_rel_h = nn.Parameter(torch.randn((2*H-1, dk), requires_grad=True)).to(device)
    rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), key_rel_h, W, H, nh, "h")
    return rel_logits_h, rel_logits_w


class AttentionCNN(nn.Module):
  def __init__(self):
    super().__init__()
    
    self.conv1 = nn.Conv2d(3, 128, 3) 
    self.pool = nn.MaxPool2d(2,2)
    self.bn1 = nn.BatchNorm2d(128)
    
    self.conv2 = nn.Conv2d(128, 64, 3)
    self.bn2 = nn.BatchNorm2d(64)
    
    self.conv3 = nn.Conv2d(64, 32, 3)
    self.bn3 = nn.BatchNorm2d(32)
    
    self.attnConv1 = attentiveConv(32, 16, 3, dk=2*16, dv=int(0.5*16), nh=8, relative=True)
    self.bn4 = nn.BatchNorm2d(16)
    
    self.fc1 = nn.Linear(16*6*6, 1024)
    self.fc2 = nn.Linear(1024, 10)
    
  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.bn1(x)
    x = self.pool(F.relu(self.conv2(x)))
    x = self.bn2(x)
    x = self.pool(F.relu(self.conv3(x)))
    x = self.bn3(x)
    x = F.relu(self.attnConv1(x))
    x = self.bn4(x)
    
    x = x.view(x.size()[0], -1)
    
    x = self.fc1(x)
    x = self.fc2(x)
    return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_classes = 10
batch_size = 32
img_width = 64
img_height = 64

model = AttentionCNN().to(device)
output = model(torch.rand([batch_size, 3, img_width, img_height]).to(device))
print(output.size())