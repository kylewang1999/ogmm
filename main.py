import logging, os, torch, numpy as np, torch, torch.nn as nn

from models.gmmreg import GMMReg
from train import *
from configs.cfgs import get_parser
from datasets.dataloader import data_loader
from datasets.modelnet import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

args = get_parser().parse_args([
    '--root', '~/Desktop/data/',
    '--batch_size', '48',
    '--epochs', '50',
    '--lr', '5e-5', '--exp_name', 'foo',
])


# Init model
model = GMMReg(args.emb_dims, args.n_clusters, args).cuda()
model = nn.DataParallel(model)
print("Let's use", torch.cuda.device_count(), "GPUs!")


train_loader, test_loader = data_loader(args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150, 200], gamma=0.1)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

exp_path = os.path.join(args.model_path, f'checkpoints/{args.exp_name}')
optim_path = os.path.join(args.model_path, f'checkpoints/{args.exp_name}/models/optim_model.pt')
log_path = os.path.join(exp_path, 'train.log')
if not os.path.exists(exp_path): 
    os.makedirs(exp_path)
    os.makedirs(os.path.join(exp_path, 'models'))   # save models at checkpoints

handler = logging.FileHandler(log_path, encoding='UTF-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.addHandler(console)

optimal_rot, optimal_tra, optimal_ccd, optimal_pcab, optimal_recall =\
    np.inf, np.inf, np.inf, np.inf, -np.inf


if os.path.exists(optim_path):
    try:
        logger.info('Loading optimizer state from {}'.format(optim_path))
        model.load_state_dict(torch.load(optim_path))
    except Exception as e:
        model.module.load_state_dict(torch.load(optim_path))
    
for epoch in range(args.epochs):
    train_metrics = train_one_epoch(epoch, model, train_loader, optimizer, logger, exp_path)
    val_metrics = eval_one_epoch(epoch, model, test_loader, logger)
    if optimal_pcab > val_metrics['pcab_dist']:
        optimal_rot = val_metrics['r_mae']
        optimal_tra = val_metrics['t_mae']
        optimal_pcab = val_metrics['pcab_dist']
        optimal_ccd = val_metrics['clip_chamfer_dist']
        optimal_recall = val_metrics['n_correct']
        save_model(model, optim_path)
        logger.info('Current best rotation: {:.04f}, transl: {:.04f}, ccd: {:.04f}, recall: {:.04f}'.format(
        optimal_rot, optimal_tra, optimal_ccd, optimal_recall))
    scheduler.step()
logger.debug('train, end')
logger.debug('done (PID=%d)', os.getpid())