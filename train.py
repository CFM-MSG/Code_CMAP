




import os
import shutil
import time
import pickle
import torch
from loss import ContrastiveLoss_aaai
from option import parser, verify_input_args
import data
from vocab import Vocabulary
from cmap import CMAP
from loss import LossModule
from evaluate import validate
from logger import AverageMeter
import logging
os.environ["CUDA_VISIBLE_DEVICES"] = "1"





logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
										datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

def resume_from_ckpt_saved_states(args, model, optimizer):



	assert os.path.isfile(args.ckpt), f"(ckpt) File not found: {args.ckpt}"
	ckpt = torch.load(args.ckpt)
	print(f"Loading file {args.ckpt}.")


	if torch.cuda.is_available():
		model.load_state_dict(ckpt['model'])
	else:
		state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)['model']
		model.load_state_dict(state_dict)
	print("Model: resume from provided state.")




	optimizer.load_state_dict(ckpt['optimizer'])
	for state in optimizer.state.values():
		for k, v in state.items():
			if torch.is_tensor(v):
				state[k] = v
				if torch.cuda.is_available():
					state[k] = state[k].cuda()
	print("Optimizer: resume from provided state.")


	best_score = ckpt['best_score']
	print("Best score: obtained from provided state.")

	return model, optimizer, best_score






def train_model(epoch, data_loader, model, criterion, criterion_contraloss_7, criterion_contraloss_14, criterion_contraloss_28, optimizer, args):


	model.train()


	loss_info = AverageMeter(precision=8)

	max_itr = len(data_loader)
	for itr, data in enumerate(data_loader):


		img_src, txt, txt_len, img_trg, _, _ = data

		if torch.cuda.is_available():
			img_src, img_trg, txt, txt_len = img_src.cuda(), img_trg.cuda(), txt.cuda(), txt_len.cuda()


		scores, scores_14, scores_28, A_IS_all_t, A_IS_all_t_14, A_IS_all_t_28, A_EM_all_t, A_EM_all_t_14, A_EM_all_t_28, A_IS_all_t_rev, A_IS_all_t_14_rev, A_IS_all_t_28_rev,  A_EM_all_t_rev, A_EM_all_t_14_rev, A_EM_all_t_28_rev, m, Tr_m, A_EM_t, A_is_rev_t, A_EM, A_IS, A_is_rev = model.forward_broadcast(img_src, img_trg, txt, txt_len)

		if args.learn_temperature:
			scores *= model.temperature.exp()
			scores_14 *= model.temperature.exp()
			scores_28 *= model.temperature.exp()











		loss = 1.0 * criterion(scores) + 1.0 * criterion(scores_14) + 1.0 * criterion(scores_28)



		loss_contra_all = 1.0 * criterion_contraloss_7(A_IS_all_t, A_IS_all_t_rev, m, Tr_m) + 1.0 * criterion_contraloss_14(A_IS_all_t_14, A_IS_all_t_14_rev, m, Tr_m) \
						  + 0.5 * criterion_contraloss_28(A_IS_all_t_28, A_IS_all_t_28_rev, m, Tr_m)


		loss = loss + 1.0 * loss_contra_all





		loss_info.update(loss.item())


		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


		if itr > 0 and (itr % args.log_step == 0 or itr + 1 == max_itr):
			log_msg = 'loss: %s' % str(loss_info)
			logging.info('[%d][%d/%d] %s' %(epoch, itr, max_itr, log_msg))

	return loss_info.avg






def validate_model(model, args, vocab, epoch=-1, best_score=None, split='val'):


	model.eval()

	with torch.no_grad():
		start = time.time()
		message, val_mes = validate(model, args, vocab, split=split)
		end = time.time()

	log_msg = "[%s][%d] >> EVALUATION <<" % (args.exp_name, epoch)
	log_msg += "\nProcessing time : %f" % (end - start)
	log_msg += message

	if best_score:
		log_msg += '\nCurrent best score: %.2f' %(best_score)

	logging.info(log_msg)

	return val_mes

def update_best_score(new_score, old_score, is_higher_better=True):
	if not old_score:
		score, updated = new_score, True
	else:
		if is_higher_better:
			score = max(new_score, old_score)
			updated = new_score > old_score
		else:
			score = min(new_score, old_score)
			updated = new_score < old_score
	return score, updated

def save_ckpt(state, is_best, args, filename='ckpt.pth', split='val'):
	ckpt_path = os.path.join(args.ckpt_dir, args.exp_name, filename)
	torch.save(state, ckpt_path)
	if is_best:
		model_best_path =  os.path.join(args.ckpt_dir, args.exp_name, split, 'model_best.pth')
		shutil.copyfile(ckpt_path, model_best_path)
		logging.info('Updating the best model checkpoint: {}'.format(model_best_path))






def main():


	args = verify_input_args(parser.parse_args())
	print(args)


	vocab_path = os.path.join(args.vocab_dir, f'{args.data_name}_vocab.pkl')
	assert os.path.isfile(vocab_path), '(vocab) File not found: {vocab_path}'
	vocab = pickle.load(open(vocab_path, 'rb'))





	model = CMAP(vocab.word2idx, args)
	print("Model version:", args.model_version)


	if torch.cuda.is_available():
		model = model.cuda()
		torch.backends.cudnn.benchmark = True


	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	

	best_score = {split:None for split in args.validate}
	if args.ckpt:
		model, optimizer, best_score = resume_from_ckpt_saved_states(args, model, optimizer)

		for split in args.validate:
			print("\nValidating on the {} split.".format(split))
			with torch.no_grad():
				_ = validate_model(model, args, vocab, -1, best_score[split], split=split) 


	criterion_bcloss = LossModule(args)
	criterion_contraloss = ContrastiveLoss_aaai(args, margin=0.6)
	criterion_contraloss_14 = ContrastiveLoss_aaai(args, margin=0.4)
	criterion_contraloss_28 = ContrastiveLoss_aaai(args, margin=0.3)

	trn_loader = data.get_train_loader(args, vocab)


	for epoch in range(args.num_epochs):


		# if epoch == 0:
		# 	for g in optimizer.param_groups:
		# 		g['lr'] = 0.00025
		if epoch != 0 and epoch < 20 and epoch % args.step_lr == 0:
			for g in optimizer.param_groups:
				print("Learning rate: {} --> {}\n".format(g['lr'], g['lr']*args.gamma_lr))
				g['lr'] *= args.gamma_lr
		if epoch != 0 and epoch >= 20 and epoch % (args.step_lr // 2) == 0:
			for g in optimizer.param_groups:
				print("Learning rate: {} --> {}\n".format(g['lr'], g['lr']*args.gamma_lr))
				g['lr'] *= args.gamma_lr

		train_model(epoch, trn_loader, model, criterion_bcloss, criterion_contraloss, criterion_contraloss_14, criterion_contraloss_28, optimizer, args)


		for split in args.validate:
			print("Validating on the {} split.".format(split))


			with torch.no_grad():
				val_score = validate_model(model, args, vocab, epoch, best_score[split], split=split)


			best_score[split], updated = update_best_score(val_score, best_score[split])


			save_ckpt({
				'args': args,
				'epoch': epoch,
				'best_score': best_score,
				'model': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			}, updated, args, split=split)

		print("")

if __name__ == '__main__':
	main()