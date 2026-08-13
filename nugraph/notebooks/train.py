# python train.py -c config.yaml > training_output.log 2>&1
import os, sys
sys.path.append('/home/hnayak/.conda/envs/NugraphBase/lib/python3.10/site-packages')
from pathlib import Path
import nugraph as ng
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import argparse
import yaml

pl.seed_everything(42, workers=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

def main(CONFIG, arg):
    ckpt_dir = CONFIG['ckpt_dir']
    input_files = CONFIG['input_files']
    run_domain_adaptation = CONFIG['run_domain_adaptation']
    fnc_name = CONFIG['train']['da_loss'] #dann, mmd, semantic, sinkhorn, None (if DA off)
    print(f"running DA: {run_domain_adaptation}")  
    subname = fnc_name.upper() if run_domain_adaptation else "OFF"
    print(f"subname: {subname}")
    
    if input_files == 1:
       os.environ["NUGRAPH_LOG"]=f'{ckpt_dir}NoDA/'
    else:
       os.environ["NUGRAPH_LOG"]=f'{ckpt_dir}DA-{subname}'
    now = datetime.now()
    folder_name = "uBooNE-Tiny-Data-%s" % now.strftime("%Y-%m-%d-%H-%M-%S")
    print(folder_name)
    logdir = Path(os.environ["NUGRAPH_LOG"])
    print("logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    logger = TensorBoardLogger(save_dir=logdir,name=folder_name) 
    callbacks = [ 
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(monitor="loss/val", mode="min"),
    ]
    
    trainer_epochs = CONFIG['train']['epochs']
    data_batch_size = CONFIG['train']['train_batch_size']
    norm_mode = CONFIG['train']['norm_mode'] # separate, shared
    
    run_event_decoder = CONFIG['run_event_decoder']
    run_semantic_decoder = CONFIG['run_semantic_decoder']
    run_filter_decoder = CONFIG['run_filter_decoder']  
    
    if run_domain_adaptation:
       model_warmup_epochs = 1 #0 if input_files == 1 else 10
    else:
       model_warmup_epochs = trainer_epochs

    Data  = ng.data.NuGraphDataModule
    Model = ng.models.NuGraph3

    source_filename= CONFIG['source_filename']
    target_filename= CONFIG['target_filename']

    if input_files == 1:
        nudata = Data(model=Model, batch_size=data_batch_size, data_source_path=source_filename)
        model_da_loss_fnc_name = None
    elif input_files == 2:
        nudata = Data(model=Model, batch_size=data_batch_size, data_source_path=source_filename, data_target_path=target_filename)
        model_da_loss_fnc_name = fnc_name

    print(f"model_da_loss_fnc_name: {model_da_loss_fnc_name}")
    nudata.event_classes = ['cc_nue', 'cc_numu', 'cc_nutau', 'nc']
    print(nudata)
    print(nudata.semantic_classes)
    print(nudata.event_classes)

    nugraph = Model(in_features=CONFIG['model']['in_feat'],  
            hit_features=CONFIG['model']['hit_feat'],
            nexus_features=CONFIG['model']['nexus_feat'],
            instance_features=CONFIG['model']['instance_feat'],
            interaction_features=CONFIG['model']['interaction_feat'],
            semantic_classes=nudata.semantic_classes, 
            event_classes=nudata.event_classes,
            num_iters=5,
            event_head=run_event_decoder,
            semantic_head=run_semantic_decoder,
            filter_head=run_filter_decoder,
            vertex_head=CONFIG['run_vertex_head'],
            instance_head=CONFIG['run_instance_head'],
            use_checkpointing=True,
            lr=CONFIG['train']['lr'],
            da_loss_fnc_name=model_da_loss_fnc_name,
            warmup_epochs=model_warmup_epochs,
            norm_mode=norm_mode)

    device_number = CONFIG['device_number']
    accelerator, devices = ng.util.configure_device(device_number)
    print(accelerator)
    print(devices)
    trainer = pl.Trainer(accelerator=accelerator,
                     devices=devices,
                     max_epochs=trainer_epochs,
                     logger=logger,
                     callbacks=callbacks,
                     deterministic=True)
    if arg.resume:
        trainer.fit(nugraph, datamodule=nudata, ckpt_path = CONFIG['resume_ckpt_path'])
        trainer.test(datamodule=nudata)
        return

    if arg.inference_only:
        checkpoint_file = CONFIG['train_ckpt_path']
        trainer.test(model=nugraph, datamodule=nudata, ckpt_path=checkpoint_file)
        return

    trainer.fit(nugraph, datamodule=nudata)
    trainer.test(datamodule=nudata)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NuGraph Training and Inference Script")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the YAML configuration file")
    parser.add_argument('--resume', action='store_true', help="Flag to resume training from the checkpoint specified in YAML")
    parser.add_argument('--inference_only', action='store_true',help="Flag to skip training and only run testing on the checkpoint")

    args = parser.parse_args()

    with open(args.config, "r") as file:
        CONFIG = yaml.safe_load(file)
        
    main(CONFIG, args)