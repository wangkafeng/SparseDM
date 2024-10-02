import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.type = "sparse"

    config.train = d(
        n_steps=500000,
        batch_size=128,
        mode='uncond',
        log_interval=10,
        eval_interval=5000,
        save_interval=50000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0002,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2500
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        embed_dim=8192,
        depth=1,
        num_heads=8,
        mlp_ratio=2,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='cifar10',
        path='assets/datasets/cifar10',
        random_flip=True,
    )

    config.sample = d(
        sample_steps=50,
        n_samples=256,
        mini_batch_size=64,
        algorithm='euler_maruyama_sde',
        path=''
    )

    return config
