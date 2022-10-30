

import os 
import cv2
import numpy as np
import copy
import json

import torch
from torch.autograd import grad

import data
from vocab import Vocabulary
from utils import params_require_grad
from cmap import CMAP
from evaluate import load_model, compute_and_process_compatibility_scores
from option import parser, verify_input_args







ONLY_BEST_RESULTS = True


NUMBER_OF_EXAMPLES = 5





NUMBER_OF_MAIN_COEFF = 3





def main_generate_heatmaps(args, model, vocab):


	categories = args.name_categories if ("all" in args.categories) else args.categories

	for category in categories:


		if ONLY_BEST_RESULTS:


			opt = copy.deepcopy(args)
			if args.study_per_category and (args.number_categories > 1):
				opt.categories = category


			queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, args.studied_split)


			studied_indices, rank_of_GT = find_best_results(model, opt, queries_loader, targets_loader)


			d = {studied_indices[i]: int(rank_of_GT[i]) for i in range(len(studied_indices))}
			directory = os.path.join(args.heatmap_dir, args.exp_name)
			if not os.path.isdir(directory):
				os.makedirs(directory)
			with open(os.path.join(directory, "metadata.json"), "a") as f:
				f.write(f"\n\nCategory: {category} \n")
				json.dump(d, f)
			print(f"Saving metadata (studied data indices, rank of GT) at {os.path.join(directory, 'metadata.json')}.")

		else:
			studied_indices = None




		opt = copy.deepcopy(args)
		if args.study_per_category and (args.number_categories > 1):
			opt.categories = category


		opt.batch_size = 1
		triplet_loader = data.get_train_loader(opt, vocab, split=args.studied_split, shuffle=False)


		generate_heatmaps_from_dataloader(triplet_loader, model, opt,
								studied_indices=studied_indices)


def find_best_results(model, opt, queries_loader, targets_loader):



	model.eval()


	with torch.no_grad():
		rank_of_GT = compute_and_process_compatibility_scores(queries_loader, targets_loader,
												model, opt, output_type="metrics")


	data_ids = rank_of_GT.sort()[1][:NUMBER_OF_EXAMPLES]	

	return data_ids.tolist(), rank_of_GT[data_ids].tolist()


def generate_heatmaps_from_dataloader(data_loader, model, args,
							studied_indices=None):



	model.eval()
	params_require_grad(model.txt_enc, False)
	

	data_loader_iterator, itr = iter(data_loader), 0
	while itr < NUMBER_OF_EXAMPLES:


		img_src, txt, txt_len, img_trg, ret_caps, data_id = next(data_loader_iterator)
		example_number = data_id[0]

		if (studied_indices is None) or (example_number in studied_indices):


			generate_heatmap_from_single_data(args, model, img_src, txt, txt_len,
												img_trg, example_number)


			formated_caption, ref_identifier, trg_identifier = data_loader.dataset.get_triplet_info(example_number)
			directory = os.path.join(args.heatmap_dir, args.exp_name, str(example_number))
			with open(os.path.join(directory, "metadata.txt"), "a") as f:
				f.write(f"{example_number}*{formated_caption}*{ref_identifier}*{trg_identifier}\n")


			del img_src, txt, txt_len, img_trg, ret_caps, data_id


			itr += 1


def generate_heatmap_from_single_data(args, model, img_src, txt, txt_len,
										img_trg, example_number):


	if torch.cuda.is_available():
		img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()

	img_src = img_src.requires_grad_(True)
	img_trg = img_trg.requires_grad_(True)


	_ = model.forward_save_intermediary(img_src, img_trg, txt, txt_len)


	heatmap_from_score(args, model, example_number,
						"IS", "A_IS_r", "A_IS_t", img_trg,
						img_src=img_src, r_is_involved=True)
	heatmap_from_score(args, model, example_number,
						"EM", "Tr_m", "A_EM_t", img_trg)


def heatmap_from_score(args, model, example_number, s_name,
							query_contrib_name, t_contrib_name,
							img_trg, img_src=None, r_is_involved=False):


	r_heatmap_tmp = None, None
	t_heatmap_tmp = None, None


	if r_is_involved:
		r_activation = model.hold_results["r_activation"]
	t_activation = model.hold_results["t_activation"]


	query_contrib = model.hold_results[query_contrib_name]
	t_contrib = model.hold_results[t_contrib_name]
	main_coeffs = get_main_coeffs(query_contrib, t_contrib)


	for main_coeff in main_coeffs:
		

		score_contrib = (query_contrib*t_contrib)[:,main_coeff]


		if r_is_involved:
			r_weights = get_weights(model, score_contrib, r_activation)
		t_weights = get_weights(model, score_contrib, t_activation)





		if r_is_involved:
			r_heatmap_tmp = (r_activation * r_weights.view(args.batch_size, -1, 1, 1)).sum(dim=1).detach().cpu()
		t_heatmap_tmp = (t_activation * t_weights.view(args.batch_size, -1, 1, 1)).sum(dim=1).detach().cpu()

		save_heatmaps(args,
						example_number,
						f"{s_name}_coeff_{main_coeff}",
						round(score_contrib[0].item(), 4),
						t_heatmap_tmp,
						img_trg,
						r_heatmap_tmp,
						img_src,
						r_is_involved)

	if r_is_involved:
		del r_activation, r_weights, r_heatmap_tmp
	del t_activation, t_weights, score_contrib, t_heatmap_tmp
	del query_contrib, t_contrib, main_coeffs


def save_heatmaps(args, example_number, s_name, s_value, t_heatmap, img_trg,
					r_heatmap=None, img_src=None, r_is_involved=False):



	if r_is_involved:
		r_heatmap = normalize_heatmap(r_heatmap)
	t_heatmap = normalize_heatmap(t_heatmap)


	directory = os.path.join(args.heatmap_dir, args.exp_name, str(example_number))
	if not os.path.isdir(directory):
		os.makedirs(directory)

	filename = os.path.join(directory, '{}_heatmap.jpg')
	if r_is_involved:
		merge_heatmap_on_image(r_heatmap, img_src, filename.format(f"{s_name}_on_src"))
	merge_heatmap_on_image(t_heatmap, img_trg, filename.format(f"{s_name}_on_trg"))


	with open(os.path.join(directory, "metadata.txt"), "a") as f:
		f.write(f"{example_number}*{s_name}*{s_value}\n")


def get_weights(model, output, conv_activation):


	model.zero_grad()

	gradients = grad(outputs=output, inputs=conv_activation, retain_graph=True)[0]

	weights = gradients.mean(dim=[2,3])
	return weights


def normalize_heatmap(heatmap):
	heatmap = torch.clamp(heatmap, 0)
	heatmap = normalize_image(heatmap)
	return heatmap


def normalize_image(img):
	if isinstance(img, torch.Tensor):
		img -= torch.min(img)
		maxi_value = torch.max(img)
	if isinstance(img, np.ndarray):
		img -= np.min(img)
		maxi_value = np.max(img)
	img /= maxi_value if maxi_value > 0 else 0.00001
	return img


def merge_heatmap_on_image(heatmap, initial_img, produced_img_path):





	heatmap = heatmap[0].data.numpy()
	initial_img = normalize_image(np.float32(initial_img.cpu()[0].permute(1, 2, 0).data.numpy()))





	heatmap = cv2.resize(heatmap, (initial_img.shape[0], initial_img.shape[1]))
	heatmap = np.uint8(255 * heatmap)
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255.
	superimposed_img = heatmap + initial_img
	superimposed_img = normalize_image(superimposed_img)
	superimposed_img = np.uint8(255 * superimposed_img)
	cv2.imwrite(produced_img_path, superimposed_img)
	print("Interpretable image registred at : {}".format(produced_img_path))


def get_main_coeffs(term_1, term_2):

	return (term_1 * term_2).detach().cpu()[0].sort()[1][-NUMBER_OF_MAIN_COEFF:]






if __name__ == '__main__':

	args = verify_input_args(parser.parse_args())


	args, model, vocab = load_model(args)


	main_generate_heatmaps(args, model, vocab)