import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook

from mmpose.core import (DistEvalHook, EvalHook, Fp16OptimizerHook, ParamwiseOptimizerHook,
                         build_optimizers)
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    # import pdb
    # pdb.set_trace()
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # determine wether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)



#######################
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    # model_without_ddp = model
    # param_dicts = [
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if
    #                    match_name_keywords(n, ['transformer']) and p.requires_grad],
    #         "lr": 0.001,
    #     },
    #     # 剩下的upsample
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, ['transformer']) and
    #              p.requires_grad],
    #         "lr": 0.01,
    #     },
    #
    # ]
    # # optimizer = make_optimizer(cfg, model, num_gpu)
    # optimizer = torch.optim.AdamW(param_dicts, lr=0.01,
    #                               weight_decay=1e-5)



    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)

    # with open('net_param.txt', 'wt') as f:
    #     [print(n, file=f) for n, p in model.named_parameters() if "backbone" in n]
    #
    # import pdb
    # pdb.set_trace()
    #
    # trans_count = 0
    # rest_count = 0
    # import pdb
    # pdb.set_trace()
    #
    # for i in range(len(optimizer.param_groups)):
    #     if optimizer.param_groups[i]['lr'] == 0.0004:
    #         trans_count += 1
    #     if optimizer.param_groups[i]['lr'] == 0.004:
    #         rest_count += 1
    #
    # print(trans_count, rest_count)
    # import pdb
    # pdb.set_trace()
    # # len(aa)==171
    # aa = [n for n, p in model.named_parameters() if 'transformer' in n]
    #######################


    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            paramwise_cfg = cfg.optimizer_config.pop('paramwise_cfg', None)
            # import pdb
            # pdb.set_trace()
            optimizer_config = ParamwiseOptimizerHook(paramwise_cfg, **cfg.optimizer_config)
            # optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            drop_last=False,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
