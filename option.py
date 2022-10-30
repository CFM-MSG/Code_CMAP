
import os
import argparse
from config import MAIN_DIR, VOCAB_DIR, CKPT_DIR, RANKING_DIR, HEATMAP_DIR

print("Main directory (default root for vocabulary files, model checkpoints, ranking files, heatmaps...): "+MAIN_DIR)

parser = argparse.ArgumentParser()


parser.add_argument('--data_name', default='fashion200K', choices=('fashionIQ', 'fashion200K', 'shoes'), help='Dataset name (fashionIQ|fashion200K|shoes).')
parser.add_argument('--vocab_dir', default=VOCAB_DIR, help='Path to saved vocabulary pickle files')


parser.add_argument('--exp_name', default='try', help='Experiment name, used as sub-directory to save experiment-related files (model, ranking files, heatmaps...).')
parser.add_argument('--ckpt_dir', default=CKPT_DIR, help='Directory in which to save the models from the different experiments.')
parser.add_argument('--ranking_dir', default=RANKING_DIR, type=str, help='Directory in which to save the ranking/prediction files, if any to save.')
parser.add_argument('--heatmap_dir', default=HEATMAP_DIR, type=str, help='Directory in which to save the heatmaps.')


parser.add_argument('--batch_size', default=128, type=int, help='Size of a mini-batch.')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
parser.add_argument('--workers', default=8, type=int, help='Number of data loader workers.')
parser.add_argument('--categories', default='all', type=str, help='Names of the data categories to consider for a given dataset. Category names must be separated with a space. Specify "all" to consider them all (the interpretation of "all" depends on the dataset).')


parser.add_argument('--model_version', default='CMAP', choices=("CMAP"), help='Model version')
parser.add_argument('--ckpt', default='', type=str, metavar='PATH', help='Path of the ckpt to resume from')
parser.add_argument('--embed_dim', default=512, type=int, help='Dimensionality of the final text & image embeddings.')
parser.add_argument('--cnn_type', default='resnet50', help='The CNN used as image encoder.')
parser.add_argument('--load_image_feature', default=0, type=int, help="")
parser.add_argument('--txt_enc_type', default='bigru', choices=('bigru', 'lstm'), help="The text encoder (bigru|lstm).")
parser.add_argument('--lstm_hidden_dim', default=1024, type=int, help='Number of hidden units in the LSTM.')
parser.add_argument('--wemb_type', default='glove', choices=('glove', 'None'), type=str, help='Word embedding (glove|None).')
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.')


parser.add_argument('--num_epochs', default=16, type=int, help='Number of training epochs.')
parser.add_argument('--lr', default=.0005, type=float, help='Initial learning rate.')
parser.add_argument('--step_lr', default=4, type=int, help="Step size, number of epochs after which to apply a learning rate decay.")
parser.add_argument('--gamma_lr', default=0.5, type=float, help="Learning rate decay.")
parser.add_argument('--learn_temperature', default=True, help='Whether to use and learn the temperature parameter (the scores given to the loss criterion are multiplied by a trainable version of --temperature).')
parser.add_argument('--temperature', default=2.65926, type=float, help='Temperature parameter.')
parser.add_argument('--validate', default='test', choices=('val', 'test', 'test-val'), help='Split(s) on which the model should be validated (if 2 are given, 2 different checkpoints of model_best will be kept, one for each validating split).')
parser.add_argument('--log_step', default=50, type=int, help='Every number of steps the log will be printed.')
parser.add_argument('--img_finetune', default=False, help='Fine-tune CNN image encoder.')
parser.add_argument('--txt_finetune', default=False, help='Fine-tune the word embeddings.')


parser.add_argument('--gradcam', action='store_true', help='Keep gradients & activations computed while encoding the images to further interprete what the network uses to make its decision.')
parser.add_argument('--studied_split', default="val", help="Split to be used for the computation (this does not impact the usual training & validation pipeline, but impacts other scripts (for evaluation or visualizations purposes)).")


def verify_input_args(args):

	ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
	if not os.path.isdir(args.ckpt_dir):
		print('Creating a directory: {}'.format(ckpt_dir))
		os.makedirs(ckpt_dir)


	args.validate = args.validate.split('-')
	for split in args.validate:
		ckpt_val_dir = os.path.join(ckpt_dir, split)
		if not os.path.isdir(ckpt_val_dir):
			print('Creating a directory: {}'.format(ckpt_val_dir))
			os.makedirs(ckpt_val_dir)


	if not os.path.isdir(args.ranking_dir):
		os.makedirs(args.ranking_dir)




	if args.wemb_type == "None":
		args.wemb_type = None


	args.name_categories = [None]
	args.recall_k_values = [1, 10, 50]
	args.recall_subset_k_values = None
	args.study_per_category = False
	if args.data_name == 'fashionIQ':
		args.name_categories = ["dress", "shirt", "toptee"]
		args.recall_k_values = [10, 50]
		args.study_per_category = True
	args.number_categories = len(args.name_categories)

	return args