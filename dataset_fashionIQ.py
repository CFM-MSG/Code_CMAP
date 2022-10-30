
import os
import itertools
import json as jsonmod

from dataset import MyDataset
from config import FASHIONIQ_IMAGE_DIR, FASHIONIQ_ANNOTATION_DIR

class FashionIQDataset(MyDataset):


	def __init__(self, split, vocab, transform, what_elements='triplet', load_image_feature=0,
					fashion_categories='all', ** kw):

		MyDataset.__init__(self, split, FASHIONIQ_IMAGE_DIR, vocab, transform=transform, what_elements=what_elements, load_image_feature=load_image_feature, ** kw)

		fashion_categories = ['dress', 'shirt', 'toptee'] if fashion_categories=='all' else sorted(fashion_categories.split())


		image_id2name_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'image_splits', f'split.{fc}.{split}.json') for fc in fashion_categories]
		image_id2name = [self.load_file(a) for a in image_id2name_files]
		self.image_id2name = list(itertools.chain.from_iterable(image_id2name))

		if self.what_elements in ["query", "triplet"]:
			prefix = 'pair2cap' if split=='test' else 'cap'
			annotations_files = [os.path.join(FASHIONIQ_ANNOTATION_DIR, 'captions', f'{prefix}.{fc}.{split}.json') for fc in fashion_categories]
			annotations = [self.load_file(a) for a in annotations_files]
			self.annotations = list(itertools.chain.from_iterable(annotations))


	def __len__(self):
		if self.what_elements=='target':
			return len(self.image_id2name)
		return 2*len(self.annotations)


	def load_file(self, f):

		with open(f, "r") as jsonfile:
			ann = jsonmod.loads(jsonfile.read())
		return ann




	def get_triplet(self, idx):


		index = idx // 2
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)


		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".png"
		img_src = self.get_transformed_image(path_src)

		path_trg = ann['target'] + ".png"
		img_trg = self.get_transformed_image(path_trg)

		return img_src, text, img_trg, real_text, idx


	def get_query(self, idx):


		index = idx // 2
		cap_slice = slice(2, None, -1) if idx%2 else slice(0, 2)


		ann = self.annotations[index]

		capts = ann['captions'][cap_slice]
		text, real_text = self.get_transformed_captions(capts)

		path_src = ann['candidate'] + ".png"
		img_src = self.get_transformed_image(path_src)
		img_src_id = self.image_id2name.index(ann['candidate'])

		img_trg_id = [self.image_id2name.index(ann['target'])]

		return img_src, text, img_src_id, img_trg_id, real_text, idx


	def get_target(self, index):

		img_id = index
		path_img = self.image_id2name[index] + ".png"
		img = self.get_transformed_image(path_img)

		return img, img_id, index




	def get_triplet_info(self, index):

		index = index // 2
		ann = self.annotations[index]
		return " [and] ".join(ann["captions"]), ann["candidate"], ann["target"]