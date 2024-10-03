import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb


#    add from diffusion_distiller
import math
def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise Exception()
    return betas


import numpy as np
beta_0 = beta_min=0.1
beta_1 = beta_max=20
def squared_diffusion_integral(s, t):  # \int_s^t beta(tau) d tau
    return beta_0 * (t - s) + (beta_1 - beta_0) * (t ** 2 - s ** 2) * 0.5

def skip_alpha(s, t):  # alpha_{t|s}, E[xt|xs]=alpha_{t|s}**0.5 xs
    x = -squared_diffusion_integral(s, t)
    return x.exp()

def skip_beta(s, t):  # beta_{t|s}, Cov[xt|xs]=beta_{t|s} I
    return 1. - skip_alpha(s, t)

def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts

# def cum_beta(t):  # the variance of xt|x0
#     raise NotImplementedError
def cum_beta(t):
    return skip_beta(0, t)

# def cum_alpha(t):
#     raise NotImplementedError
def cum_alpha(t):
    return skip_alpha(0, t)

def marginal_prob(x0, t):  # the mean and std of q(xt|x0)
    alpha = cum_alpha(t)
    beta = cum_beta(t)
    mean = stp(alpha ** 0.5, x0)  # E[xt|x0]
    std = beta ** 0.5  # Cov[xt|x0] ** 0.5
    return mean, std

def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)

def duplicate(tensor, *size):
    return tensor.unsqueeze(dim=0).expand(*size, *tensor.shape)

import torch.nn.functional as F
def inference(net_, x, t, x0):
    print('enter inference ')
    time_scale = 1
    # return net_(x, t * time_scale, **extra_args)
    # return net_(x, t * time_scale)
    # return net_(x, t * 999, **kwargs) 

    # x0 = _batch
    t_init=0
    t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
    mean, std = marginal_prob(x0, t)
    eps = torch.randn_like(x0)
    xt = mean + stp(std, eps)

    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    t = t.to(xt.device)
    if t.dim() == 0:
        t = duplicate(t, xt.size(0))
    return net_(x, t * 999)
    # return net_(x, t)


def E_(input, t, shape):
    # out = torch.gather(input, 0, t)
    out = torch.gather(input.to('cuda:0'), dim=0, index=t.to('cuda:0'))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)
    return out

def get_alpha_sigma(x, t):
    betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=50)  #    1024
    # print('betas = ', betas)
    
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    alpha = E_(sqrt_alphas_cumprod, t, x.shape)
    sigma = E_(sqrt_one_minus_alphas_cumprod, t, x.shape)
    return alpha, sigma


def distill_loss(student_diffusion, x, t, extra_args=None, eps=None, student_device=None):
    gamma=0.3
    time_scale=1

    print('enter distill_loss ...')
    print('t = ', t)
    if eps is None:
        eps = torch.randn_like(x)   #    noise
    with torch.no_grad():
        alpha, sigma = get_alpha_sigma(x, t + 1)   #     error ?
        z = alpha * x + sigma * eps
        # alpha_s, sigma_s = student_diffusion.get_alpha_sigma(x, t // 2)  #    modify
        alpha_s, sigma_s = get_alpha_sigma(x, t // 2) 
        alpha_1, sigma_1 = get_alpha_sigma(x, t)
        v = inference(student_diffusion, z.float(), t.float() + 1, x0=x).double()
        rec = (alpha * z - sigma * v).clip(-1, 1)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        v_1 = inference(z_1.float(), t.float(), extra_args).double()
        x_2 = (alpha_1 * z_1 - sigma_1 * v_1).clip(-1, 1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        v_2 = alpha_s * eps_2 - sigma_s * x_2
        if gamma == 0:
            w = 1
        else:
            w = torch.pow(1 + alpha_s / sigma_s, gamma)
    v = student_diffusion.net_(z.float(), t.float() * time_scale, **extra_args)
    my_rec = (alpha_s * z - sigma_s * v).clip(-1, 1)
    return F.mse_loss(w * v.float(), w * v_2.float())


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator()   #    
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    dataset = get_dataset(**config.dataset)
    print('dataset.fid_stat = ', dataset.fid_stat)
    assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    # train_state = utils.initialize_train_state(config, device)  #    modify
    # train_state = utils.load_train_state(config, device)   #    for pretrain
    # train_state = utils.load_train_state_c10(config, device) 
    # train_state = utils.load_train_state_c10_pr(config, device) 
    # train_state = utils.load_train_state_c10_pd2(config, device)    #    from  pretrained
    train_state = utils.load_train_state_c10_ls(config, device)
    # nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
    #     train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)  #     for accelerator
    teacher_nnet_ema, nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.teacher_nnet, train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)  
    # nnet, nnet_ema, optimizer, train_dataset_loader = train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader 

    lr_scheduler = train_state.lr_scheduler
    # train_state.resume(config.ckpt_root)  #    close for test

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()


    # set the score_model to train
    score_model= sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())    #    this is like teacher model ??
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())   #    modify
    # score_model = sde.ScoreModelPD(nnet, teacher_nnet, pred=config.pred, sde=sde.VPSDE())
    # score_model_ema = sde.ScoreModelPD(nnet_ema, teacher_nnet_ema, pred=config.pred, sde=sde.VPSDE())
    # score_model_teacher = sde.ScoreModelPD(teacher_nnet, pred=config.pred, sde=sde.VPSDE())
    # score_model_ema_teacher = sde.ScoreModelPD(teacher_nnet_ema, pred=config.pred, sde=sde.VPSDE())



    def train_step(_batch):
        _metrics = dict()
        optimizer.zero_grad()
        if config.train.mode == 'uncond':  #    run this
            loss = sde.LSimple(score_model, _batch, pred=config.pred)   #    
        elif config.train.mode == 'cond':
            loss = sde.LSimple(score_model, _batch[0], pred=config.pred, y=_batch[1])
        elif config.train.mode == 'pd':   #    add for PD
            
            # loss = sde.LSimple(score_model, _batch, pred=config.pred)   #     must change this 
            '''
            # loss = teacher_diffusion.distill_loss(student_diffusion, img, time, extra_args)  #    change from diffusion_distiller/train_utils.py
            betas = make_beta_schedule("cosine", cosine_s=8e-3, n_timestep=50).to(device)   #    sde.py/euler_maruyama/sample_steps = 50
            print('betas = ', betas)
            betas = betas.type(torch.float64)
            num_timesteps = int(betas.shape[0])
            print('num_timesteps = ', num_timesteps)   #  DDPM 1024 , uvit 50 
            print('_batch.shape[0] = ', _batch.shape[0])  # 1
            time = 2 * torch.randint(0, num_timesteps, (_batch.shape[0],), device=device)   # 84 ?
            print('time = ', time)   
            loss = distill_loss(nnet, _batch, time)
            '''

            #    from sde.LSimple, run pass
            
            x0 = _batch
            t, noise, xt = score_model.sde.sample(x0)    #    run VPSDE .  This must use teacher model ?  not random sample ?
            # replace score_model.noise_pred
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            t = t.to(xt.device)
            if t.dim() == 0:
                t = duplicate(t, xt.size(0))
            pred = nnet(xt, t * 999)  #    modify to teacher 
            noise_pred = pred

            # loss = mos(noise - noise_pred)  #    like p_losses . 

            target = teacher_nnet_ema(xt, t * 999)   #    modify for distillation_loss_simple ,  loss down fast , fid why ?
            # target = teacher_nnet(xt, t)   #    loss down fast 
            # target = teacher_nnet(x0, t * 999)
            # target = teacher_nnet(x0, t)
            # loss = mos(target - pred)

            # loss = 0.01 * mos(noise - noise_pred) + mos(target - pred)   # from bs=128 loss number
            # loss = mos(noise - noise_pred) + mos(target - pred)
            loss = mos(noise - noise_pred) + 0.1 * mos(target - pred)

            #    from sde.LSimple and sde.sample
            '''
            x0 = _batch

            # t, noise, xt = score_model.sde.sample(x0)
            t_init=0
            t = torch.rand(x0.shape[0], device=x0.device) * (1. - t_init) + t_init
            mean, std = marginal_prob(x0, t)
            eps = torch.randn_like(x0)   #   
            xt = mean + stp(std, eps)
            t, noise, xt = t, eps, xt

            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t)
            t = t.to(xt.device)
            if t.dim() == 0:
                t = duplicate(t, xt.size(0))
            pred = nnet(xt, t * 999)
            noise_pred = pred

            loss = mos(noise - noise_pred)
            '''
        else:
            raise NotImplementedError(config.train.mode)
        _metrics['loss'] = accelerator.gather(loss.detach()).mean()
        accelerator.backward(loss.mean())
        if 'grad_clip' in config and config.grad_clip > 0:
            accelerator.clip_grad_norm_(nnet.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        train_state.ema_update(config.get('ema_rate', 0.9999))
        train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    #    
    def eval_step(n_samples, sample_steps, algorithm):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}, algorithm={algorithm}, '
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            _x_init = torch.randn(_n_samples, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            elif config.train.mode == 'pd':   #    add for PD
                kwargs = dict()
            else:
                raise NotImplementedError

            if algorithm == 'euler_maruyama_sde':
                return sde.euler_maruyama(sde.ReverseSDE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'euler_maruyama_ode':
                return sde.euler_maruyama(sde.ODE(score_model_ema), _x_init, sample_steps, **kwargs)
            elif algorithm == 'dpm_solver':
                noise_schedule = NoiseScheduleVP(schedule='linear')
                model_fn = model_wrapper(
                    score_model_ema.noise_pred,
                    noise_schedule,
                    time_input_type='0',
                    model_kwargs=kwargs
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule)
                return dpm_solver.sample(
                    _x_init,
                    steps=sample_steps,
                    eps=1e-4,   #    ?
                    adaptive_step_size=False,
                    fast_version=True,
                )
            else:
                raise NotImplementedError

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()
    

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')
    print('train_state.step = ', train_state.step)
    print('config.train.n_steps = ', config.train.n_steps)

    # import copy
    # tearcher_model = copy.deepcopy(nnet)    #     add for PD

    # start training
    # tearcher_model.eval()
    # for p in tearcher_model.parameters():
    #     p.requires_grad_(False)        #     noly train  student.

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        metrics = train_step(batch)

        nnet.eval()   #    ??
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        #    
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            logging.info('Save a grid of images...')
            x_init = torch.randn(100, *dataset.data_shape, device=device)
            if config.train.mode == 'uncond':
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=x_init, sample_steps=50)    #    inference mask error ??
            elif config.train.mode == 'cond':
                y = einops.repeat(torch.arange(10, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=x_init, sample_steps=50, y=y)
            elif config.train.mode == 'pd':
                samples = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=x_init, sample_steps=50)   #     add for PD
            else:
                raise NotImplementedError
            samples = make_grid(dataset.unpreprocess(samples), 10)
            save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            print('save_image = ', os.path.join(config.sample_dir, f'{train_state.step}.png'))   #    
            wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()


        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))   #     169M 
            accelerator.wait_for_everyone()
            #    
            fid = eval_step(n_samples=10000, sample_steps=50, algorithm='dpm_solver')  # calculate fid of the saved checkpoint
            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    #     close for no step_fid
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))   #     load pretrained model 
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps, algorithm=config.sample.algorithm)   #    for inference mask error ??



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
