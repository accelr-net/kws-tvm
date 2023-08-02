import torch
import torch.nn.functional as F
import torchaudio
import sys
import pickle
import argparse
import librosa
from torchvision.transforms import ToTensor
import numpy as np
import time
import pprint
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

from tvm.contrib import graph_executor

# For auto tuning purpose
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm

from torchaudio.datasets import SPEECHCOMMANDS
import os

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("../training/", download=False)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


def load_model(mdl_path: str):

    try:
        model = torch.jit.load(mdl_path)
        print("Model loaded successfully.")
        print("\n")
        return model
    
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("\n")
        return None


def load_labels(lbl_path: str):
    with open(lbl_path, 'rb') as handle:
        lable = pickle.load(handle)
        return lable


def transform_audio(audio, sample_rate: int):
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                              n_fft=1024,
                                                              win_length=None,
                                                              hop_length=512,
                                                              n_mels=128,
                                                              power=2.0)
    transformed = transform(audio)
    return transformed

def get_shape():
    set = SubsetSC("testing")
    waveform, sample_rate, *_ = set[0]
    spectogram = mel_conversion(waveform,sample_rate)
    return spectogram.shape

def reshape(waveform,sample_rate):
    if waveform.shape[-1] < sample_rate :
        waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
    elif waveform.shape[-1] > sample_rate:
        waveform = waveform[:,:sample_rate]
    waveform = transform_audio(waveform, sample_rate)
    return waveform

def mel_conversion(waveform,sample_rate):
    waveform = reshape(waveform,sample_rate)
    spectogram = ToTensor()(librosa.power_to_db(waveform.squeeze().numpy(), ref=np.max))
    spectogram = spectogram.unsqueeze(0)
    return spectogram

def main(args):

    # Loading the Pytorch Model (CUDA/CPU) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n")
    print(" ----------------------------")
    print("| Loading Pre Trained Model |")
    print(" ----------------------------")
    print("\n")

    model = load_model(args.model_path)
    

    labels = load_labels(args.lbl_path)

    CLASS_TO_IDX = {c: i for i, c in enumerate(labels)}
    idx_to_class = {v: k for k, v in CLASS_TO_IDX.items()}

    # Insertion of data set for inference
    

    test_set = SubsetSC("testing")
    waveforms = []
    sample_rates = []
    labels = []
    speaker_ids = []
    utterance_numbers = []
    print(" ------------------------")
    print("| Starting Optimization |")
    print(" ------------------------")
    print("\n")

    # Importing the ML Graph to Relay (TVM)
    input_shape = get_shape()
    input_name = "input0"
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(model, shape_list)
    target = tvm.target.Target("llvm") # Optimization will happen to target device given here.


    ## Auto tuning Optimization

    number = 10 # specifies the number of different configurations
    repeat = 1 # how many measurements we will take of each configuration
    min_repeat_ms = 0  # how long need to run configuration test. If the number of repeats falls under this time, it will be increased , since we're tuning on a CPU, can be set to 0
    timeout = 10  # in seconds, upper limit on how long to run training code for each tested configuration

    # create a TVM runner
    runner = autotvm.LocalRunner(
        number=number,
        repeat=repeat,
        timeout=timeout,
        min_repeat_ms=min_repeat_ms,
        enable_cpu_cache_flush=True,
    )

    tuning_option = {
    "tuner": "xgb",
    "trials": 20,
    "early_stopping": 100, # the minimum number of trails to run before a condition that stops the search 
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner #indicates where trial code will be built, and where it will be run
    ),
    "tuning_records": "kws-rn18-autotuning.json",
    }

    print("** Extracting Tasks.... ")
    print("\n")

    # begin by extracting the tasks from the ML model
    tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
    pprint.pprint(tasks)




    print("\n")
    print("** Tuning Extracted Tasks.... ")
    print("\n")

    # Tune the extracted tasks sequentially.
    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        tuner_obj = XGBTuner(task, loss_type="rank") #pairwise rank loss to train cost model relative rank score
        tuner_obj.tune(
            n_trial=min(tuning_option["trials"], len(task.config_space)),
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                autotvm.callback.log_to_file(tuning_option["tuning_records"]),
            ],
        )

    # Relay building with the tuned Model
    with autotvm.apply_history_best(tuning_option["tuning_records"]):
        with tvm.transform.PassContext(opt_level=3, config={}):
            lib = relay.build(mod, target=target, params=params)
    dev = tvm.cpu(0)


    print("\n")
    print(" -----------------------")
    print("| Running Inferencing |")
    print(" -----------------------")
    # Performance Variables
    count = 0
    pcount = 0
    correct = 0
    wrong = 0 
    pycorrect = 0
    pywrong = 0

    # Executing the graph on the implemented Device
    m = graph_executor.GraphModule(lib["default"](dev))

    #Performance Measurement of the Data set

    #Inferencing with Pytorch and it's performance measurement

    pstart_time = time.time()

    for t in test_set:
        waveform_t, sample_rate_t, label_t, speaker_id_t, utterance_number_t = t
        spectogram_t = mel_conversion(waveform_t,sample_rate_t)

        
        pred = torch.argmax(model(spectogram_t), dim=1)
        pytorch_pred = idx_to_class[pred.item()]

        pcount = pcount + 1

        if(pytorch_pred == label_t):
            pycorrect = pycorrect + 1
        else:
            pywrong = pywrong + 1 
        
    pyaccuracy = pycorrect/pcount

    print("\n")
    print("----------- Inferencing performance with Pytorch  ------------")
    print("\n")
    print("Count: " + str(pcount) + " Correct: " + str(pycorrect) + " Wrong: " + str(pywrong))
    print("Accuaracy: " + str(pyaccuracy))
    print('Inference speed: %.2f samples/s'%(pcount/(time.time()-pstart_time)))
    print("\n")

    #Inferencing with ML network implemented in llvm optimized via TVM

    start_time = time.time()

    for l in test_set:

        waveform_t, sample_rate_t, label_t, speaker_id_t, utterance_number_t = l
        spectogram_t = mel_conversion(waveform_t,sample_rate_t)

        input_name_t = "input0" 
        m.set_input(input_name_t, tvm.nd.array(spectogram_t))
        m.run()
        tvm_out = m.get_output(0)
        top1_tvm = np.argmax(tvm_out.numpy()[0])
        top1_tvm_label = idx_to_class[top1_tvm]

        count = count + 1
        if(top1_tvm_label == label_t):
            correct = correct + 1
        else:
            wrong = wrong + 1 


    accuracy = correct/count
    
    print("----------- Inferencing performance with ML network implemented in llvm optimized via TVM  ------------")
    print("\n")
    print("Count: " + str(count) + " Correct: " + str(correct) + " Wrong: " + str(wrong))
    print("Accuaracy: " + str(accuracy))
    print('Inference speed: %.2f samples/s'%(count/(time.time()-start_time)))
    print("\n")

    

    

    
   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_path", help = "The model path for pytorch kws model", default="./checkpoints/resnet18-kws-best-acc.pt")
    parser.add_argument("-l", "--lbl_path", help=" The path of label pickle file", default="./lable.pickle")
    parser.add_argument("-s", "--sample_rate", help="The resize audio sample rate", default=8000)

    args = parser.parse_args()
    if not all([args.model_path, args.lbl_path, args.sample_rate]):
        parser.print_help()
        print("\nThese arguments are required: --model_path --lbl_path and --sample_rate")
        sys.exit(1)
    main(args)

# python --model_path resnet18-kws-best-acc.pt --lbl_path --sample_rate
