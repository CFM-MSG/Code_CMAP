
import torch
import torch.nn as nn
import torch.nn.functional as F
def l2norm(x):
	norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
	return torch.div(x, norm)

class LossModule(nn.Module):

	def __init__(self, opt):
		super(LossModule, self).__init__()

	def forward(self, scores):




		GT_labels = torch.arange(scores.shape[0]).long()
		GT_labels = torch.autograd.Variable(GT_labels)
		if torch.cuda.is_available():
			GT_labels = GT_labels.cuda()


		loss = F.cross_entropy(scores, GT_labels, reduction = 'mean')

		return loss

def compute_l2(x1, x2):
	l2_loss = torch.nn.MSELoss(reduction='sum')
	return l2_loss(x1, x2)

class ContrastiveLoss(nn.Module):

	def __init__(self, opt, margin=0):
		super(ContrastiveLoss, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_is_t, A_em_t, m, tr_m):

		batch_size = m.size(0)



		scores_is = (m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)





		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)














		return cost_s.sum(0)

class ContrastiveLoss_aaai(nn.Module):

	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_aaai, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_is_t, A_is_t_rev, m, tr_m):

		batch_size = m.size(0)



		scores_is = (m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * A_is_t_rev).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_is_t_rev).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (1.0 * diagonal_is + 0.0 * diagonal_is_trm)
		diagonal_em_all = (0.0 * diagonal_em_trm + 1.0 * diagonal_em)





		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)














		return cost_s.sum(0)















		return cost_s.sum(0)

class ContrastiveLoss_aaai_att(nn.Module):

	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_aaai_att, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_is, A_is_rev, A_em, tr_m):

		batch_size = tr_m.size(0)








		A_em = A_em.view(batch_size, 1, self.embed_dim)
		A_is = A_is.view(batch_size, 1, self.embed_dim)
		A_is_rev = A_is_rev.view(batch_size, 1, self.embed_dim)
		scores_is = (A_em.repeat(1, batch_size, 1) * A_is.repeat(1, batch_size, 1)).sum(-1)

		scores_em = (A_em.repeat(1, batch_size, 1) * A_is_rev.repeat(1, batch_size, 1)).sum(-1)

		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)





		diagonal_is_all = diagonal_is
		diagonal_em_all = diagonal_em





		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)
















		return cost_s.sum(0)


class ContrastiveLoss_aaai_emrev(nn.Module):

	def __init__(self, opt, margin=0):
		super(ContrastiveLoss_aaai_emrev, self).__init__()
		self.opt = opt
		self.embed_dim = opt.embed_dim
		self.margin = margin


	def forward(self, A_em_t, A_em_t_rev, m, tr_m):

		batch_size = m.size(0)



		scores_is = (m.view(batch_size, 1, self.embed_dim) * A_em_t_rev).sum(-1)
		scores_em = (m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		scores_is_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t_rev).sum(-1)
		scores_em_trm = (tr_m.view(batch_size, 1, self.embed_dim) * A_em_t).sum(-1)
		diagonal_is = scores_is.diag().view(batch_size, 1)
		diagonal_em = scores_em.diag().view(batch_size, 1)
		diagonal_is_trm = scores_is_trm.diag().view(batch_size, 1)
		diagonal_em_trm = scores_em_trm.diag().view(batch_size, 1)
		diagonal_is_all = (0.4 * diagonal_is + 0.6 * diagonal_is_trm)
		diagonal_em_all = (0.6 * diagonal_em_trm + 0.4 * diagonal_em)





		cost_s = (self.margin + diagonal_is_all - diagonal_em_all).clamp(min=0)














		return cost_s.sum(0)