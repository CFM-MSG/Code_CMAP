


import os
import argparse
import pickle
from collections import Counter
import json
import nltk
from config import VOCAB_DIR, FASHIONIQ_ANNOTATION_DIR, SHOES_ANNOTATION_DIR, CIRR_ANNOTATION_DIR, FASHION200K_ANNOTATION_DIR, cleanCaption








ANNOTATIONS = {
  'fashionIQ': [f'{FASHIONIQ_ANNOTATION_DIR}/captions/cap.{fc}.train.json' for fc in ['dress','shirt','toptee']],
  'shoes': [f'{SHOES_ANNOTATION_DIR}/triplet.train.json'],
  'fashion200K': [f'{FASHION200K_ANNOTATION_DIR}/{fc}_train_detect_all.txt' for fc in ['dress', 'skirt', 'jacket', 'pants', 'top']]
}






class Vocabulary(object):


  def __init__(self):
    self.idx = 0
    self.word2idx = {}
    self.idx2word = {}

  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __call__(self, word):
    return self.word2idx.get(word, self.word2idx['<unk>'])

  def __len__(self):
    return len(self.word2idx)






def from_fashionIQ_json(p):
  with open(p, "r") as jsonfile:
    ann = json.loads(jsonfile.read())
  captions = [cleanCaption(a["captions"][0]) for a in ann]
  captions += [cleanCaption(a["captions"][1]) for a in ann]
  return captions


def from_shoes_json(p):
  with open(p, "r") as jsonfile:
    ann = json.loads(jsonfile.read())
  captions = [cleanCaption(a["RelativeCaption"]) for a in ann]
  return captions




def from_fashion200K_txt(p):
    with open(p, 'r') as file:
        content = file.read().splitlines()

    caption = [cleanCaption(line.split('\t')[-1]) for line in content]
    return caption


def from_txt(txt):
  captions = []
  with open(txt, 'rb') as f:
    for line in f:
      captions.append(line.strip())
  return captions






def build_vocab(data_name, threshold=0):



  counter = Counter()


  for p in ANNOTATIONS[data_name]:
    if data_name == 'fashionIQ':
      captions = from_fashionIQ_json(p)
    elif data_name == 'shoes':
      captions = from_shoes_json(p)
    elif data_name == 'fashion200K':
      captions = from_fashion200K_txt(p)
    else:
      captions = from_txt(p)


    for caption in captions:
      tokens = nltk.tokenize.word_tokenize(caption.lower())
      counter.update(tokens)



  words = [word for word, cnt in counter.items() if cnt >= threshold]
  print('Vocabulary size: {}'.format(len(words)))


  vocab = Vocabulary()
  vocab.add_word('<pad>')
  vocab.add_word('<start>')
  vocab.add_word('<and>')
  vocab.add_word('<end>')
  vocab.add_word('<unk>')

  if data_name == 'fashion200K':
      vocab.add_word('replace')
      vocab.add_word('with')


  for word in words:
    vocab.add_word(word)

  return vocab


def main(data_name, threshold, vocab_dir):

  vocab = build_vocab(data_name, threshold=threshold)

  if not os.path.isdir(vocab_dir):
    os.makedirs(vocab_dir)

  vocab_path = os.path.join(vocab_dir, f'{data_name}_vocab.pkl')
  with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
  print("Saved vocabulary file to ", vocab_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name', default='fashion200K', choices=('fashionIQ', 'shoes', 'fashion200K'), help='Name of the dataset for which to build the vocab (fashionIQ|shoes|fashion200K)')
  parser.add_argument('--vocab_dir', default=VOCAB_DIR, help='Root directory for the vocab files.')
  parser.add_argument('--threshold', default=0, type=int, help="Minimal number of occurrences for a word to be included in the vocab.")
  opt = parser.parse_args()
  main(opt.data_name, opt.threshold, opt.vocab_dir)
