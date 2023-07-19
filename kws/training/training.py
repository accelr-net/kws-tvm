import argparse
import torch

import pickle
import os
import sys
import matplotlib as plt
from pytorch_lightning import  Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsummary import summary
from resnet18_pl import KWSModel
from dataset_pl import KWSDataModule


def main(args):

    
    
    CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
                'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
                'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        
        # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    

    try:
        model = KWSModel(num_classes=args.num_classes, epochs=args.max_epochs, lr=args.lr)
        summary(model, (1, 128, 63))
        
    except Exception as e:
        print(f"Error creating DModel object: {e}")
        sys.exit(1)


    example_input = torch.randn(1, 1, 128, 63)
    print(example_input.shape)

    try:
        datamodule = KWSDataModule(batch_size=args.batch_size, num_workers=args.num_workers,
                                n_fft=args.n_fft, n_mels=args.n_mels,
                                win_length=args.win_length, hop_length=args.hop_length,
                                class_dict=CLASS_TO_IDX)
        datamodule.setup()

    except Exception as e:
        print(f"Error creating Data Module object: {e}")
        sys.exit(1)

    
    print(model)
 

        

    model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(args.path, "checkpoints"),
            filename="resnet18-kws-best-acc",
            save_top_k=1,
            verbose=True,
            monitor='test_acc',
            mode='max',
        )
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    trainer = Trainer(accelerator=args.accelerator,
                        devices=args.devices,
                        precision=args.precision,
                        max_epochs=args.max_epochs,
                        logger= None,
                        callbacks=[model_checkpoint])
    model.hparams.sample_rate = datamodule.sample_rate
    model.hparams.idx_to_class = idx_to_class
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    

    model = model.load_from_checkpoint(os.path.join(
    args.path, "checkpoints", "resnet18-kws-best-acc.ckpt"))
    model.eval()
    script = model.to_torchscript(method="trace", example_inputs=example_input)

    # Define example inputs
    #  # Replace with appropriate input shape

# Trace the model
    #traced_model = torch.jit.trace(model, example_input)


    
    

    # save for use in production environment
    model_path = os.path.join(args.path, "checkpoints",
                            "resnet18-kws-best-acc.pt")
    torch.jit.save(script, model_path)


    labels = CLASSES

    with open('filename.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        # model training hyperparameters
        parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--max-epochs', type=int, default=2, metavar='N',
                            help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001)')

        # where dataset will be stored
        parser.add_argument("--path", type=str, default="./")

        # 35 keywords + silence + unknown
        parser.add_argument("--num-classes", type=int, default=35)
    
        # mel spectrogram parameters
        parser.add_argument("--n-fft", type=int, default=1024)
        parser.add_argument("--n-mels", type=int, default=128)
        parser.add_argument("--win-length", type=int, default=None)
        parser.add_argument("--hop-length", type=int, default=512)

        # 16-bit fp model to reduce the size
        parser.add_argument("--precision", default=32)
        parser.add_argument("--accelerator", default='cpu')
        parser.add_argument("--devices", default=1)
        parser.add_argument("--num-workers", type=int, default=8)

        parser.add_argument("--no-wandb", default=False, action='store_true')

        args = parser.parse_args()

        main(args)
