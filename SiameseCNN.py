import torch 
import random 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset 

class myDatasetName(Dataset):
	'''
		Custom siamese dataset.
		Input: 
			data: list (image path, id)
			transforms: image transforms 
		Return: 
			img1: Left image 
			img2: Right image
			label: 0 if images are from different classes, 1 if they're not.
	'''
	def __init__(self, data, transforms=None):
		self.data = data
		self.transforms = transforms

	def __getitem__(self, index):
		record1 = random.choice(self.data)
		# Decide randomly if return a pair of images of the same class or not,
		same = random.randint(0, 1)
		if same:
			while True:
				record2 = random.choice(self.data)
				if record1[1] == record2[1]:
					label = 1
					break
		else: 
			while True: 
				record2 = random.choice(self.data)
				if record1[1] != record2[1]:
					label = 0
					break
		# Read images
		img1 = Image.open(record1[0])
		img2 = Image.open(record2[0])
		# Convert label to tensor 
		label = torch.from_numpy(np.array([label], dtype=np.float32))
		# Apply transforms
		if self.transforms is not None:
			img1 = self.transforms(img1)
			img2 = self.transforms(img2)

		return img1, img2, label

	def __len__(self):
		return len(self.data)


class siameseCNN(nn.Module):
	'''
		It returns two feature vectors to find the similarity between the inputs.
		Input: 
			img1, img2: 105 x 105px grayscale image (depth = 1)
	'''
	def __init__(self):
		super().__init__()
		self.feature_maps = nn.Sequential(nn.Conv2d(1, 64, 10),  
                             	 nn.ReLU(inplace=True),
                             	 nn.MaxPool2d(2,2), 
                             
                             	 nn.Conv2d(64, 128, 7),
                             	 nn.ReLU(inplace=True),
                             	 nn.MaxPool2d(2,2), 
                             
                             	 nn.Conv2d(128, 128, 4),
                             	 nn.ReLU(inplace=True),
                             	 nn.MaxPool2d(2,2), 
                             
                             	 nn.Conv2d(128, 256, 4),
                             	 nn.ReLU(inplace=True))

		self.feature_vects = nn.Sequential(nn.Linear(6*6*256, 4096),
                             	 nn.Dropout(0.3),
                             	 nn.Sigmoid(),
                             
                            	 nn.Linear(4096, 16))

	def forward_once(self, x):
		x = self,feature_maps(x)
		x = x.view(x.size()[0], -1)
		x = self.feature_vects(x)
		return x

	def forward(self, x1, x2):
		featvect1 = self.forward_once(x1)
		featvect2 = self.forward_once(x2)
		return featvect1, featvect2 


def contrastive_loss(featvect1, featvect2, label):
	'''
		Learns discrimitaive features for images.
	'''
	euclidean_distance = F.pairwise_distance(featvect1, featvect2)
	loss = torch.mean(1-label) * torch.pow(euclidean_distance, 2) + 
		  (label) * torch.pow(torch.clamp(2. - euclidean_distance, min=0.0), 2)
	return loss


def inference(img1, img2, threshold=0.5):
	'''
		Defines if im1 is of the same class than img2 or not.
		Input: 
			img1, img2: 105 x 105px grayscale images
			threshold: value between 0 and 1 defining how similar the 
			images should be to be considered as same.
		Output:
			1 if same class, otherwise 0
	'''
	out1, out2 = siameseCNN(img1, img2)
	euclidean_distance = F.pairwise_distance(out1, out2)
	if (1 - euclidean_distance.item()) > threshold:
		return 1 
	return 0  
