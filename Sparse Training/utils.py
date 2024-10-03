import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
# from apex.contrib.sparsity import ASP  #   


def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        print('uvit = ', UViT(**kwargs))
        return UViT(**kwargs)           #   
    elif name == 'uvit_sparse':   #    add
        from libs.uvit_sparse import UViT
        return UViT(**kwargs)   
    elif name == 'uvit_mask':   #    add,  SparseConv, SparseLinear
        from libs.uvit_mask import UViT
        print('uvit_mask = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_linear':   #    add, SparseLinear
        from libs.uvit_linear import UViT
        print('uvit_linear = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step':   #    add, step SparseLinear   step15
        from libs.uvit_step import UViT
        print('uvit_step = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step2':   #    add, step SparseLinear
        from libs.uvit_step2 import UViT
        print('uvit_step2 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step3':   #    add, step SparseLinear
        from libs.uvit_step3 import UViT
        print('uvit_step3 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step5':   #    add, step SparseLinear  5:8
        from libs.uvit_step5 import UViT
        print('uvit_step5 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step4':   #    add, step SparseLinear
        from libs.uvit_step4 import UViT
        print('uvit_step4 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step7':   #    add, step SparseLinear
        from libs.uvit_step7 import UViT
        print('uvit_step7 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step8':   #    add, step SparseLinear
        from libs.uvit_step8 import UViT
        print('uvit_step8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step15':   #    add, step SparseLinear
        from libs.uvit_step15 import UViT
        print('uvit_step15 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step16':   #    add, step SparseLinear
        from libs.uvit_step16 import UViT
        print('uvit_step16 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step31':   #    add, step SparseLinear
        from libs.uvit_step31 import UViT
        print('uvit_step31 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step32':   #    add, step SparseLinear
        from libs.uvit_step32 import UViT
        print('uvit_step32 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step8_8':   #    add, dense weight  for multi-scale mask inference
        from libs.uvit_step8_8 import UViT
        print('uvit_step8_8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_scale':   #    add, dense weight  for multi-scale mask inference
        from libs.uvit_scale import UViT
        print('uvit_scale = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_multi_scales':   #    add, dense weight  for multi-scale mask inference, only change sparse_ops_multi_scales
        from libs.uvit_multi_scales import UViT
        print('uvit_multi_scales = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_multi_scales2':   #    add, dense weight  for multi-scale mask inference, change 3 files
        from libs.uvit_multi_scales2 import UViT
        print('uvit_multi_scales2 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_multi_scales3':   #    add, dense weight  for multi-scale mask inference, change 3 files
        from libs.uvit_multi_scales3 import UViT
        print('uvit_multi_scales3 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_dense8_8':   #    add, train dense weight  for multi-scale mask inference
        from libs.uvit_dense8_8 import UViT
        print('uvit_dense8_8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_78':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_78 import UViT
        print('uvit_eval_78 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_58':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_58 import UViT
        print('uvit_eval_58 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_48':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_48 import UViT
        print('uvit_eval_48 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_multi':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_multi import UViT
        print('uvit_eval_multi = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_multi_78':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_multi_78 import UViT
        print('uvit_eval_multi_78 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_multi_68':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_multi_68 import UViT
        print('uvit_eval_multi_68 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_eval_multi_58':   #    add, dense weight  for multi-scale mask inference   8:8
        from libs.uvit_eval_multi_58 import UViT
        print('uvit_eval_multi_58 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step1_1':   #    add, dense weight  for multi-scale mask inference
        from libs.uvit_step1_1 import UViT
        print('uvit_step1_1 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_dense1_1':   #    add, train dense weight  for multi-scale mask inference
        from libs.uvit_dense1_1 import UViT
        print('uvit_dense1_1 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step6_8':   #    add, step SparseLinear
        from libs.uvit_step6_8 import UViT
        print('uvit_step6_8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step5_8':   #    add, step SparseLinear
        from libs.uvit_step5_8 import UViT
        print('uvit_step5_8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    elif name == 'uvit_step4_8':   #    add, step SparseLinear
        from libs.uvit_step4_8 import UViT
        print('uvit_step4_8 = ', UViT(**kwargs))
        return UViT(**kwargs)   
    # elif name == 'uvit_asp':   #    add
    #     from libs.uvit_sparse import UViT
    #     model = UViT(**kwargs)
    #     ASP.prune_trained_model(model, optimizer)
    #     return    model
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        print('uvit = ', UViT(**kwargs))
        return UViT(**kwargs)
    elif name == 'uvit_t2i_linear':   #    add, SparseLinear
        from libs.uvit_t2i_linear import UViT
        print('uvit = ', UViT(**kwargs))
        return UViT(**kwargs)
    else:
        raise NotImplementedError(name)


def set_seed(seed: int):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):   #    ?
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)  #    ??

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))    #     169M

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    #    add for asp
    def load_asp(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                print('load_asp',  os.path.join(path, f'{key}.pth'))
                if key == 'nnet_ema':
                    val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'), strict=False)  
                else:
                     val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))  #    load optimizer error ?

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


class TrainStateLS(object):
    def __init__(self, optimizer, lr_scheduler, step, teacher_nnet=None, nnet=None, nnet_ema=None):  #    modify
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.teacher_nnet = teacher_nnet  #    add
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))    #     169M

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))


    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


class TrainStatePD(object):
    def __init__(self, optimizer, lr_scheduler, step, teacher_nnet=None, teacher_nnet_ema=None,  nnet=None, nnet_ema=None):  #    modify
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.teacher_nnet = teacher_nnet  #    add
        self.teacher_nnet_ema = teacher_nnet_ema  #    add
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))    #     169M

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    #    add for asp
    # def load_asp(self, path):
    #     logging.info(f'load from {path}')
    #     self.step = torch.load(os.path.join(path, 'step.pth'))
    #     for key, val in self.__dict__.items():
    #         if key != 'step' and val is not None:
    #             print('load_asp',  os.path.join(path, f'{key}.pth'))
    #             if key == 'nnet_ema':
    #                 val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'), strict=False)  
    #             else:
    #                  val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))  #    load optimizer error ?

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)



class TrainStatePD2(object):
    def __init__(self, optimizer, lr_scheduler, step, teacher_nnet_ema=None,  nnet=None, nnet_ema=None):  #    modify
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.teacher_nnet_ema = teacher_nnet_ema  #    add
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))    #     169M

    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

    #    add for asp
    # def load_asp(self, path):
    #     logging.info(f'load from {path}')
    #     self.step = torch.load(os.path.join(path, 'step.pth'))
    #     for key, val in self.__dict__.items():
    #         if key != 'step' and val is not None:
    #             print('load_asp',  os.path.join(path, f'{key}.pth'))
    #             if key == 'nnet_ema':
    #                 val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'), strict=False)  
    #             else:
    #                  val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))  #    load optimizer error ?

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)



def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    add for ASP
    # ASP.prune_trained_model(nnet, optimizer)   # error ??
    # ASP.prune_trained_model(nnet_ema, optimizer)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


#    add for dynamic vit model
def initialize_train_state_dynamic_vit(config, device):
    params = []

    base_rate = 0.9
    SPARSE_RATIO = [base_rate, base_rate - 0.2, base_rate - 0.4]

    # from libs.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
    # from libs.dylvvit import LVViTDiffPruning, LVViT_Teacher
    from libs.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
    # from libs.dyswin import AdaSwinTransformer, SwinTransformer_Teacher
    # from  libs.unet_ddpm_sparse import UNet  
    image_size = 256
    num_classes = 1000
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass
    PRUNING_LOC = [3, 6, 9]
    KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
    print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
    # nnet = VisionTransformerDiffPruning(
    #     patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
    # pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
    # )
    nnet = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
    # nnet.image_size = [1, 3, 256, 256]
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    # nnet_ema = VisionTransformerDiffPruning(
    #     patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
    # pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
    # )
    nnet_ema = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state

def initialize_train_state_edm(config, device):
    params = []

    from  libs.networks import EDMPrecond  
    image_size = 256
    num_classes = 1000
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass
    nnet = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet.image_size = [1, 3, 256, 256]
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    nnet_ema = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state

#    add for unet_ddpm model
def initialize_train_state_ddpm(config, device):
    params = []

    from  libs.unet_ddpm_sparse import UNet  
    image_size = 256
    num_classes = 1000
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass
    nnet = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet.image_size = [1, 3, 256, 256]
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    nnet_ema = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def initialize_train_state_ddpm_dense(config, device):
    params = []

    from  libs.unet_ddpm import UNet  
    image_size = 256
    num_classes = 1000
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass
    nnet = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet.image_size = [1, 3, 256, 256]
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    nnet_ema = UNet(in_channel = 3,
        channel = 128-16,
        channel_multiplier = [1, 2, 2, 4, 4],
        n_res_blocks = 2,
        attn_strides = [8, 16],
        attn_heads = 4,
        use_affine_time = True,
        dropout = 0,
        fold = 1)
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def initialize_train_state_impr_ddpm(config, device):
    params = []

    from  libs.unet_impr_ddpm import SuperResModel, UNetModel  
    # image_size = 256
    # num_classes = 1000
    # assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    # latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass

    # if image_size == 256:
    #     channel_mult = (1, 1, 2, 2, 4, 4)
    # elif image_size == 64:
    #     channel_mult = (1, 2, 3, 4)
    # elif image_size == 32:
    channel_mult = (1, 2, 2, 2)
    num_classes = 10
    num_channels = 128
    num_res_blocks = 3
    learn_sigma = True
    dropout = 0.3
    attention_ds = [2, 4]
    class_cond =  False
    use_checkpoint =  False
    num_heads =  4
    num_heads_upsample =  -1
    use_scale_shift_norm =  True

    # else:
    #     raise ValueError(f"unsupported image size: {image_size}")
    nnet = UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
    # nnet.image_size = [1, 3, 256, 256]
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    nnet_ema = UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state

#    add for DiT model
def initialize_train_state_dit(config, device):
    params = []

    from models_dit import DiT_models, DiT
    image_size = 256
    num_classes = 1000
    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8
    # model = DiT_models['DiT-XL/2'](
    # nnet = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    from libs.uvit import UViT
    # print('uvit = ', UViT(**kwargs))
    # nnet = UViT(**config.dit)           #     run pass
    nnet= DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, input_size=latent_size, num_classes=num_classes)
    print('nnet = ', nnet)
    # nnet = get_nnet(**config.nnet)      #      modify

    params += nnet.parameters()

    # nnet_ema = DiT_models['DiT-S/2'](
    #     input_size=latent_size,
    #     num_classes=num_classes
    # )
    # nnet_ema = UViT(**config.dit)   #     run pass
    nnet_ema = DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, input_size=latent_size, num_classes=num_classes)
    # nnet_ema = get_nnet(**config.nnet)   #      modify
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


# add by   
def load_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    train_state.load('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')
    train_state.to(device)
    return train_state


# add by   
def load_train_state_asp_c10(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load_asp('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')   # error
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


# add by   
def load_train_state_asp_img64(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load_asp('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')   # error
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid/300000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


# add by   
def load_train_state_ldm_asp_img256(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load_asp('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')   # error
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large/300000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state

# add by   
def load_train_state_ckpt64(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    train_state.load('./pretrained_models/imagenet64_uvit_mid/300000.ckpt/')
    train_state.to(device)
    return train_state


# add by   
def load_train_state_ckpt256(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    train_state.load('./workdir/imagenet256_uvit_large/default/ckpts/300000.ckpt/')
    train_state.to(device)
    return train_state



# add by   
def load_train_state_c10(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_pr(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state.load('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')  # g41 error, g42 pass
    # train_state.load_asp('./workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/')
    train_state.to(device)
    return train_state


def load_train_state_c10_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd


def load_train_state_c10_pd2(config, device):
    params = []

    #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD2(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd


def load_train_state_c10_ls(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    '''
    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)

    #    add
    teacher_nnet_ckpt = teacher_nnet.load_state_dict(torch.load('./pretrained_models/cifar10_uvit_small.pth', map_location='cpu'), strict=False)
    # teacher_nnet_ckpt.to(device)
    return train_state, teacher_nnet_ckpt
    '''
    return train_state_ls


def load_train_state_c10_8_8(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_scales3.3/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_scales3.3/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_scales3.3/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_1_16(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_16/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_16/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_16/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_15_16(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_15_16/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_15_16/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_15_16/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_1_8(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_8/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_8/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_8/default/ckpts/350000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_7_8(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_7_8/default/ckpts/300000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_7_8/default/ckpts/300000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_7_8/default/ckpts/300000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_1_4(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_4/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_4/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_4/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_3_4(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_3_4/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_3_4/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_3_4/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_2_4(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls/default/ckpts/500000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls/default/ckpts/500000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls/default/ckpts/500000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_31_32(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_31_32/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_31_32/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_31_32/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_ls_1_32(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_32/default/ckpts/200000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_32/default/ckpts/200000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/cifar10_uvit_small_linear_ls_1_32/default/ckpts/200000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.to(device)
    
    return train_state_ls


def load_train_state_c10_step15(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/cifar10_uvit_small_linear_step15_100000/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_step7(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/cifar10_uvit_small_linear_step7_100000/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_step3(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/cifar10_uvit_small_linear_step3_100000/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state



def load_train_state_c10_step58(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/cifar10_uvit_small_linear_step5_100000/default/ckpts/100000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


# add by   
def load_train_state_c10_multi_step15(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet15) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet15)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state

def load_train_state_c10_multi_step7(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet7) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet7)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_multi_step3(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet3) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet3)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_multi_step2(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet2) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet2)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_multi_step6_8(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet6_8) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet6_8)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_multi_step5_8(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet5_8) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet5_8)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_c10_multi_step4_8(config, device, step):
    params = []

    # nnet = get_nnet(**config.nnet)      #      modify
    nnet = get_nnet(**config.nnet4_8) 
    # nnet = get_multi_nnet(config.nnet, **config.nnet)      #    
    params += nnet.parameters()
    # nnet_ema = get_nnet(**config.nnet)     #      modify
    nnet_ema = get_nnet(**config.nnet4_8)
    # nnet_ema = get_multi_nnet(config.nnet, **config.nnet)  
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.nnet_ema.load_state_dict(torch.load('./workdir/' +config.config_name+ '/default/ckpts/'+str(step)+'.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


# add by   
def load_train_state_cel64(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_cel64_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd


def load_train_state_cel64_ls(config, device):
    params = []

    #    add
    teacher_nnet = get_nnet(**config.teacher_nnet)
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_ls.teacher_nnet.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False)  #    
    # train_state_ls.nnet.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False) 
    # train_state_ls.nnet_ema.load_state_dict(torch.load('./pretrained_models/celeba_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_ls.to(device)
    return train_state_ls


def load_train_state_cel64_ls_2_4(config, device):
    params = []

    #    add
    teacher_nnet = get_nnet(**config.teacher_nnet)
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_ls.teacher_nnet.load_state_dict(torch.load('workdir/celeba64_uvit_small_linear_ls/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False)  #    
    train_state_ls.nnet.load_state_dict(torch.load('workdir/celeba64_uvit_small_linear_ls/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet_ema.load_state_dict(torch.load('workdir/celeba64_uvit_small_linear_ls/default/ckpts/450000.ckpt/nnet_ema.pth', map_location='cpu'), strict=False) 
    train_state_ls.to(device)
    return train_state_ls


# add by   
def load_train_state_img64(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_img642(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False) 
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_img64_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_mid.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd


# add by   
def load_train_state_img64_large(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state


def load_train_state_img64_large2(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state.nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False)
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False)  #    
    train_state.to(device)
    return train_state



def load_train_state_img64_large_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet64_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd



# add by   
def load_train_state_ldm(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state


def load_train_state_ldm2(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False)
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state


def load_train_state_ldm_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')

    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 

    # train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth'), strict=False) 
    # train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth'), strict=False)  #    
    # train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth'), strict=False) 
    # train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth'), strict=False) 

    train_state_pd.to(device)
    return train_state_pd


def load_train_state_ldm_ls(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_ls = TrainStateLS(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_ls.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')

    train_state_ls.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    train_state_ls.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 


    train_state_ls.to(device)
    return train_state_ls


# add by   
def load_train_state_ldm_huge256(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_huge.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state


def load_train_state_huge256_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')

    # train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    # train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False)  #    
    # train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 
    # train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'), strict=False) 

    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_huge.pth'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_huge.pth'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_huge.pth'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_huge.pth'), strict=False) 

    train_state_pd.to(device)
    return train_state_pd



# add by   
def load_train_state_ldm512(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet512_uvit_large.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state



# add by   
def load_train_state_coco256(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state


def load_train_state_coco256_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd



# add by   
def load_train_state_coco_deep(config, device):
    params = []

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    # train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/imagenet256_uvit_large.pth', map_location='cpu'))  #    why this pass ??   no change sparse ?
    train_state.nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small_deep.pth', map_location='cpu'), strict=False)   #    for SparseLinear
    train_state.to(device)
    return train_state



def load_train_state_coco_deep_pd(config, device):
    params = []

    teacher_nnet = get_nnet(**config.teacher_nnet)   #    add
    teacher_nnet_ema = get_nnet(**config.teacher_nnet)
    teacher_nnet_ema.eval()

    nnet = get_nnet(**config.nnet)      #    
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    #    modify
    train_state_pd = TrainStatePD(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0, teacher_nnet=teacher_nnet, teacher_nnet_ema=teacher_nnet_ema, 
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state_pd.ema_update(0)
    # train_state.load('/home/carter/THU/U-ViT/workdir/cifar10_uvit_small/default/ckpts/500000.ckpt/nnet_ema.pth')  #    add
    # train_state.load('./pretrained_models/imagenet256_uvit_large/10000.ckpt')
    train_state_pd.teacher_nnet.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small_deep.pth', map_location='cpu'), strict=False) 
    train_state_pd.teacher_nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small_deep.pth', map_location='cpu'), strict=False)  #    
    train_state_pd.nnet.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small_deep.pth', map_location='cpu'), strict=False) 
    train_state_pd.nnet_ema.load_state_dict(torch.load('./pretrained_models/mscoco_uvit_small_deep.pth', map_location='cpu'), strict=False) 
    train_state_pd.to(device)
    return train_state_pd



def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
