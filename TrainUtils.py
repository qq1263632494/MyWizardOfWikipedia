# 提供参数的工具类
import os
import shutil

import torch
from torch.utils.data import RandomSampler, DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
from AgentModel import TransformerMemNetAgent
from CommonUtils import trange, tqdm
from DataUtils import compile_all_dialogs
from FeatureUtils import Tokenizer
from FileUtils import read_json
from torch.optim import Adam


class ARG:
    def __init__(self,
                 train_batch_size=4,
                 max_steps=-1,
                 weight_decay=0.0,
                 num_train_epochs=3,
                 learning_rate=3e-5,
                 adam_epsilon=1e-8,
                 warmup_steps=0,
                 warmup_proportion=0.1,
                 gradient_accumulation_steps=1,
                 device='cpu',
                 max_grad_norm=1.0,
                 save_dir='./tmp',
                 alpha=0.95):
        self.train_batch_size = train_batch_size
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.alpha = alpha


class WizardOfWikipediaDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)


# 终极版的训练器， 再也不需要考虑不同模型的问题了， 输入为 参数:ARG, 训练数据集:Dataset， 模型
def final_train_function(args: ARG,
                         train_dataset: WizardOfWikipediaDataset,
                         agent: TransformerMemNetAgent):
    """ Train the model """
    agent.model.to(args.device)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = AdamW(agent.model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    if args.warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total
        )

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    agent.model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )

    # 清空模型缓存目录
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            agent.model.train()
            keys = batch.keys()
            inputs = {}
            for key in keys:
                inputs[key] = batch[key].to(args.device)

            loss = agent.compute_train_batch_loss(inputs, args.alpha)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(agent.model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                agent.model.zero_grad()
                global_step += 1

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
        print('\n' + str(tr_loss / global_step))
        try:
            PATH = args.save_dir + '/EP' + str(_) + '.pth'
            torch.save(agent.model, PATH)
            print('当前模型保存在：' + PATH)
        except Exception:
            print(Exception)

    return global_step, tr_loss / global_step


def build_train_dataset(tokenizer: Tokenizer, DATA_PATH: str):
    DATA = read_json(DATA_PATH)
    examples = compile_all_dialogs(DATA)

    real_examples = []
    for example in examples:
        if example.choosen_index >= 0 and len(example.knowledge_pool) >= 32:
            real_examples.append(example)

    all_data = []
    for example in real_examples:
        all_data.append(tokenizer.tokenize_example_as_train_data(example))

    dataset = WizardOfWikipediaDataset(all_data)
    return dataset


def no_schedule_train(train_dataset: WizardOfWikipediaDataset, agent: TransformerMemNetAgent, args: ARG):
    """ Train the model """
    agent.model.to(args.device)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = Adam(agent.model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    agent.model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )

    # 清空模型缓存目录
    if os.path.isdir(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.mkdir(args.save_dir)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            agent.model.train()
            keys = batch.keys()
            inputs = {}
            for key in keys:
                inputs[key] = batch[key].to(args.device)

            loss = agent.compute_train_batch_loss(inputs, args.alpha)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            global_step += 1

        print('\n' + str(tr_loss / global_step))
        try:
            PATH = args.save_dir + '/EP' + str(_) + '.pth'
            torch.save(agent.model, PATH)
            print('当前模型保存在：' + PATH)
        except Exception:
            print(Exception)

    return global_step, tr_loss / global_step
