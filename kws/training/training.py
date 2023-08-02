import argparse
import torch
import pickle
import os
import sys
import matplotlib as plt
from pytorch_lightning import  Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchsummary import summary



def main(args):

    print(sys.path)

    
    
    CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
                'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
                'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
                'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
        
        # make a dictionary from CLASSES to integers
    CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

    if not os.path.exists(args.path):
        os.makedirs(args.path, exist_ok=True)

    
    if args.model == "resnet18":

        from resnet18.res18model import Res18Model
        from resnet18.res18data import Res18Data

        try:
            model = Res18Model(num_classes=args.num_classes, epochs=args.max_epochs, lr=args.lr)
            summary(model, (1, 128, 63))
            
        except Exception as e:
            print(f"Error creating DModel object: {e}")
            sys.exit(1)


        example_input = torch.randn(1, 1, 128, 63)
        print(example_input.shape)

        try:
            datamodule = Res18Data(batch_size=args.batch_size, num_workers=args.num_workers,
                                    n_fft=args.n_fft, n_mels=args.n_mels,
                                    win_length=args.win_length, hop_length=args.hop_length,
                                    class_dict=CLASS_TO_IDX)
            datamodule.setup()

        except Exception as e:
            print(f"Error creating Data Module object: {e}")
            sys.exit(1)

    
        print(model)
    
    elif args.model == "m5":

        from m5.m5model import M5Model
        from m5.m5data import M5Data


        try:
            model = M5Model(n_input=1, n_channel=32, stride=16, n_output=35, epochs=args.max_epochs, lr=args.lr)
            summary(model, (1, 8000))
            
        except Exception as e:
            print(f"Error creating Model object: {e}")
            sys.exit(1)


        example_input = torch.randn(1,1,8000)
        

        try:
            datamodule = M5Data(batch_size=args.batch_size, num_workers=args.num_workers,
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
            filename=args.model + "-kws-best-acc",
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
    args.path, "checkpoints", args.model+"-kws-best-acc.ckpt"))
    model.eval()
    script = model.to_torchscript(method="trace", example_inputs=example_input)
   
    

    # save for use in production environment
    model_path = os.path.join(args.path, "checkpoints",
                           args.model + "-kws-best-acc.pt")
    torch.jit.save(script, model_path)


    labels = CLASSES

    with open('../rn18/lable.pickle', 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

        parser = argparse.ArgumentParser()
        # model training hyperparameters
        parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--max-epochs', type=int, default=1, metavar='N',
                            help='number of epochs to train (default: 30)')
        parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                            help='learning rate (default: 0.001)')

        # where dataset will be stored
        parser.add_argument("--path", type=str, default="../rn18/")

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

        parser.add_argument("--model", default='resnet18')

        args = parser.parse_args()

        main(args)
