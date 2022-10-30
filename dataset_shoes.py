
import os
import itertools
import json as jsonmod
import torch

from dataset import MyDataset
from config import SHOES_IMAGE_DIR, SHOES_ANNOTATION_DIR


class ShoesDataset(MyDataset):


	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0, ** kw):


		MyDataset.__init__(self, split, SHOES_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)


		self.image_id2name = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'split.{split}.json'))


		if self.what_elements in ["query", "triplet"]:
			self.annotations = self.load_file(os.path.join(SHOES_ANNOTATION_DIR, f'triplet.{split}.json'))


	def __len__(self):
		if self.what_elements=='target':
			return len(self.image_id2name)
		return len(self.annotations)


	def load_file(self, f):

		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann




	def get_triplet(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['ImageName']
		img_trg = self.get_transformed_image(path_trg)

		return img_src, text, img_trg, real_text, index


	def get_query(self, index):

		ann = self.annotations[index]

		capts = ann['RelativeCaption']
		text, real_text = self.get_transformed_captions([capts])

		path_src = ann['ReferenceImageName']
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['ReferenceImageName'])

		img_trg_id = [self.image_id2name.index(ann['ImageName'])]

		return img_src, text, img_src_id, img_trg_id, real_text, index


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index]
		img = self.get_transformed_image(path_img)

		return img, img_id, index



	def get_triplet_info(self, index):

		ann = self.annotations[index]
		return ann["RelativeCaption"], ann["ReferenceImageName"], ann["ImageName"]