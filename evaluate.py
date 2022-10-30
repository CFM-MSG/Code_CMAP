import os
import time
from tqdm import tqdm
import pickle
import copy
import torch

from option import parser, verify_input_args
import data
from vocab import Vocabulary
from cmap import CMAP



def validate(model, args, vocab, output_type="metrics", max_retrieve = 50, split='test'):

	output_type_inpractice = "rankings" if args.data_name == "" else output_type


	results = []
	categories = args.name_categories if ("all" in args.categories) else args.categories.split(' ')


	model.eval()


	for category in categories:


		opt = copy.deepcopy(args)
		if args.study_per_category and (args.number_categories > 1):
			opt.categories = category


		queries_loader, targets_loader = data.get_eval_loaders(opt, vocab, split)


		with torch.no_grad():
			start = time.time()
			res = compute_and_process_compatibility_scores(queries_loader, targets_loader,
													model, opt, output_type_inpractice,
													max_retrieve)

			end = time.time()
			print("\nProcessing time : ", end - start)


		results.append(res)

	if output_type=="metrics":



		message, val_mes = results_func(results, args)
		return message, val_mes

	return results


def compute_and_process_compatibility_scores(data_loader_query, data_loader_target,
										model, args, output_type="metrics",
										max_retrieve=50):


	nb_queries= len(data_loader_query.dataset)


	if output_type=="metrics":

		ret = torch.zeros(nb_queries, requires_grad=False)
	else:

		ret = torch.zeros(nb_queries, max_retrieve, requires_grad=False).int()


	all_img_embs, all_img_embs_14, all_img_embs_28 = compute_necessary_embeddings_img(data_loader_target, model, args)

	for data in tqdm(data_loader_query):


		_, txt, txt_len, img_src_ids, img_trg_ids, _, indices = data
		if torch.cuda.is_available():
			txt, txt_len = txt.cuda(), txt_len.cuda()


		txt_embs = model.get_txt_embedding(txt, txt_len)


		for i, index in enumerate(indices):


			txt_emb = txt_embs[i]
			img_src_id = img_src_ids[i]
			GT_indices = img_trg_ids[i]


			img_src_emb = all_img_embs[img_src_id]
			img_src_emb_14 = all_img_embs_14[img_src_id]
			img_src_emb_28 = all_img_embs_28[img_src_id]


			cs, cs_14, cs_28 = model.get_compatibility_from_embeddings_one_query_multiple_targets(
										img_src_emb, img_src_emb_14, img_src_emb_28, txt_emb, all_img_embs, all_img_embs_14, all_img_embs_28)
			cs = cs + cs_14 + cs_28

			cs[img_src_id] = float('-inf')


			cs_sorted_ind = cs.sort(descending=True)[1]
			

			if output_type == "metrics":
				ret[index] = get_rank_of_GT(cs_sorted_ind, GT_indices)[0]
			else:
				ret[index, :max_retrieve] = cs_sorted_ind[:max_retrieve].cpu().int()


	return ret


def compute_necessary_embeddings_img(data_loader_target, model, args):



	img_trg_embs = None

	for data in tqdm(data_loader_target):


		img_trg, _, indices = data
		indices = torch.tensor(indices)
		if torch.cuda.is_available():
			img_trg = img_trg.cuda()


		img_trg_emb, img_trg_emb_14, img_trg_emb_28 = model.get_image_embedding(img_trg)

		if img_trg_embs is None:
			emb_sz = [len(data_loader_target.dataset), args.embed_dim]
			img_trg_embs = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False)
			emb_sz_14 = [len(data_loader_target.dataset), args.embed_dim]
			img_trg_embs_14 = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False)
			emb_sz_28 = [len(data_loader_target.dataset), args.embed_dim]
			img_trg_embs_28 = torch.zeros(emb_sz, dtype=img_trg_emb.dtype, requires_grad=False)

			if torch.cuda.is_available():
				img_trg_embs = img_trg_embs.cuda()
				img_trg_embs_14 = img_trg_embs_14.cuda()
				img_trg_embs_28 = img_trg_embs_28.cuda()

		if torch.cuda.is_available():
			img_trg_embs[indices] = img_trg_emb
			img_trg_embs_14[indices] = img_trg_emb_14
			img_trg_embs_28[indices] = img_trg_emb_28
		else:
			img_trg_embs[indices] = img_trg_emb.cpu()
			img_trg_embs_14[indices] = img_trg_emb_14.cpu()
			img_trg_embs_28[indices] = img_trg_emb_28.cpu()

	return img_trg_embs, img_trg_embs_14, img_trg_embs_28


def get_rank_of_GT(sorted_ind, GT_indices):

	rank_of_GT = float('+inf')
	best_GT = None
	for GT_index in GT_indices:
		tmp = torch.nonzero(sorted_ind == GT_index)
		if tmp.size(0) > 0:
			tmp = tmp.item()
			if tmp < rank_of_GT:
				rank_of_GT = tmp
				best_GT = GT_index
	return rank_of_GT, best_GT


def get_recall(rank_of_GT, K):
	return 100 * (rank_of_GT < K).float().mean()


def results_func(results, args):


	nb_categories = len(results)


	H = {"r%d"%k:[] for k in args.recall_k_values}
	H.update({"medr":[], "meanr":[], "nb_queries":[]})


	for i in range(nb_categories):

		for k in args.recall_k_values:
			H["r%d"%k].append(get_recall(results[i], k))
		H["medr"].append(torch.floor(torch.median(results[i])) + 1)
		H["meanr"].append(results[i].mean() + 1)
		H["nb_queries"].append(len(results[i]))


	H["avg_per_cat"] = [sum([H["r%d"%k][i] for k in args.recall_k_values])/len(args.recall_k_values) for i in range(nb_categories)]
	val_mes = sum(H["avg_per_cat"])/nb_categories
	H["nb_total_queries"] = sum(H["nb_queries"])
	for k in args.recall_k_values:
		H["R%d"%k] = sum([H["r%d"%k][i]*H["nb_queries"][i] for i in range(nb_categories)])/H["nb_total_queries"]
	H["rsum"] = sum([H["R%d"%k] for k in args.recall_k_values])
	H["med_rsum"] = sum(H["medr"])
	H["mean_rsum"] = sum(H["meanr"])


	message = ""


	if nb_categories > 1:
		categories = args.name_categories if ("all" in args.categories) else args.categories
		cat_detail = ", ".join(["%.2f ({})".format(cat) for cat in categories])

		message += ("\nMedian rank: " + cat_detail) % tuple(H["medr"])
		message += ("\nMean rank: " + cat_detail) % tuple(H["meanr"])
		for k in args.recall_k_values:
			message += ("\nMetric R@%d: " + cat_detail) \
						% tuple([k]+H["r%d"%k])

		message += ("\nRecall average: " + cat_detail) % tuple(H["avg_per_cat"])


		message += "\nGlobal recall metrics: {}".format( \
						", ".join(["%.2f (R@%d)" % (H["R%d"%k], k) \
						for k in args.recall_k_values]))


	else:
		message += "\nMedian rank: %.2f" % (H["medr"][0])
		message += "\nMean rank: %.2f" % (H["meanr"][0])
		for k in args.recall_k_values:
			message += "\nMetric R@%d: %.2f" % (k, H["r%d"%k][0])

	message += "\nValidation measure: %.2f\n" % (val_mes)

	return message, val_mes


def load_model(args):


	vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
	assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
	vocab = pickle.load(open(vocab_path, 'rb'))



	model = CMAP(vocab.word2idx, args)
	print("Model version:", args.model_version)

	if torch.cuda.is_available():
		model = model.cuda()
		torch.backends.cudnn.benchmark = True


	if args.ckpt:


		assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
		print(f"Loading file {args.ckpt}.")

		if torch.cuda.is_available():
			model.load_state_dict(torch.load(args.ckpt)['model'])
		else :
			state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
			model.load_state_dict(state_dict)
		print("Model: resume from provided state.")

	return args, model, vocab


if __name__ == '__main__':

	args = verify_input_args(parser.parse_args())


	args, model, vocab = load_model(args)

	start = time.time()
	with torch.no_grad():
		message, _ = validate(model, args, vocab, split = args.studied_split)
	print(message)


	basename = ""
	if os.path.basename(args.ckpt) != "model_best.pth":
		basename = "_%s" % os.path.basename(os.path.basename(args.ckpt))
	save_txt = os.path.abspath( os.path.join(args.ckpt, os.path.pardir, os.path.pardir, 'eval_message%s.txt' % basename) )
	with open(save_txt, 'a') as f:
		f.write(args.data_name + ' ' + args.studied_split + ' ' + args.exp_name + '\n######')
		f.write(message + '\n######\n')

	end = time.time()
	print("\nProcessing time : ", end - start)
