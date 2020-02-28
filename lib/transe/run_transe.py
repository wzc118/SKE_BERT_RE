import os
import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    '-e',
                    type=str,
                    default='train',
                    help='train or dev')
args = parser.parse_args()

phase = args.mode
base_path = './benchmarks/transe_ske_pso' + '/' + phase + '/'

# dataloader for training
train_dataloader = TrainDataLoader(
    #in_path = "./benchmarks/transe_ske/", 
    in_path = base_path,
    nbatches = 100,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 5)


# define the model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200, 
    p_norm = 2, 
    norm_flag = True)


# define the loss function
model = NegativeSampling(
    model = transe, 
    loss = MarginLoss(margin = 5.0),
    batch_size = train_dataloader.get_batch_size()
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
if not os.path.exists(os.path.join('checkpoint',phase)):
    os.mkdir(os.path.join('checkpoint',phase))
save_path = os.path.join('checkpoint',phase,'transe_pso.ckpt')
transe.save_checkpoint(save_path)

