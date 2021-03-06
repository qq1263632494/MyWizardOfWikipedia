from AgentModel import ModelOption, TransformerMemNetAgent, TransformerMemNet
from parlai.core.dict import DictionaryAgent
from parlai.core.opt import Opt
from DataUtils import compile_all_dialogs, compile_one_dialog, display_example
from FeatureUtils import Tokenizer
import pickle
import torch
from FileUtils import read_json
from torch.utils.data import DataLoader
from TrainUtils import ARG, final_train_function, WizardOfWikipediaDataset, no_schedule_train
from CommonUtils import tqdm
with open('./dictionary.pickle', 'rb') as fi:
    dictionary = pickle.load(fi)
embedding = torch.load('./fasttext-subword-300-embedding.pytorch')
tokenizer = Tokenizer(dictionary=dictionary, pad_to_max=True, max_length=64)
DATA_PATH = './WizardOfWikipedia/train.json'
DATA = read_json(DATA_PATH)
examples = compile_all_dialogs(DATA)
real_examples = []
for example in examples:
    if example.choosen_index >= 0 and len(example.knowledge_pool) >= 32:
        real_examples.append(example)
len(real_examples)
all_data = []
for example in tqdm(real_examples):
    all_data.append(tokenizer.tokenize_example_as_train_data(example))
train_data = all_data[0:3600] + all_data[3700:55100] + all_data[55200:58700] + all_data[58800:]
dataset = WizardOfWikipediaDataset(train_data)
agent = TransformerMemNetAgent(opt=ModelOption(),
                               tokenizer=tokenizer,
                               embedding=embedding)
arg = ARG(num_train_epochs=10,
          learning_rate=5e-4,
          device="cuda:0",
          train_batch_size=16,
          print_every=500,
          warmup_proportion=0.4,
          save_dir='./WOWTMPFINAL',
          warmup_method='linear')
final_train_function(args=arg, train_dataset=dataset, agent=agent)