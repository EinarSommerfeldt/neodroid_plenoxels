import json
import numpy as np
import matplotlib.pyplot as plt
import os

def get_stats(filename):
    f = open(filename, "r", encoding="utf8")

    eval_stats = dict() #{-1 : [psnr,mse] , 0 : [psnr,mse], 1 : [psnr,mse], ....}
    train_stats = dict() #{0: [[it_0, psnr_0], [it_1, psnr_1], ...], 1:...}
    epoch = -1
    for x in f:
        if "eval stats" in x:
            json_str = x.split(":",1)[1].replace("\'","\"")
            json_object = json.loads(json_str)
            eval_stats[epoch] = [json_object["psnr"], json_object["mse"]]
            epoch += 1
        if "epoch" in x:
            line_list = x.split(" ")
            it = psnr = 0
            for elem in line_list:
                if "psnr" in elem:
                    psnr = float(elem.split("=")[1][:-1])
                if "/" in elem:
                    it = int(elem.split("/")[0])
                    break

            if epoch not in train_stats:
                train_stats[epoch] = []
            train_stats[epoch].append([it,psnr])
    return eval_stats,train_stats

"""
eval_stats: {-1 : [psnr,mse] , 0 : [psnr,mse], 1 : [psnr,mse], ....}
train_stats: {0: [[it_0, psnr_0], [it_1, psnr_1], ...], 1:...}

returns eval_psnr_max, train_psnr_max
"""
def peak_psnr(eval_stats, train_stats):
    train_psnrs = np.array([])
    eval_psnrs = np.array([])

    for epoch, values in train_stats.items():
        train_epoch = np.array(values)
        psnr = train_epoch[:,1]
        train_psnrs = np.block([train_psnrs, psnr])

    for epoch, values in eval_stats.items():
        psnr = values[0]
        eval_psnrs = np.block([eval_psnrs, psnr])

    return eval_psnrs.max(), train_psnrs.max()

#eval_stats: {-1 : [psnr,mse] , 0 : [psnr,mse], 1 : [psnr,mse], ....}
#train_stats: {0: [[it_0, psnr_0], [it_1, psnr_1], ...], 1:...}
def make_plots(title, eval_stats, train_stats):
    max_it = 0
    epochs = len(train_stats)
    train_its = np.array([])
    train_psnrs = np.array([])

    for epoch, values in train_stats.items():
        train_epoch = np.array(values)
        it = train_epoch[:,0]
        psnr = train_epoch[:,1]

        if max_it == 0:
            max_it = it[-1]
        it += max_it*epoch

        train_its = np.block([train_its, it])
        train_psnrs = np.block([train_psnrs, psnr])
    
    eval_its = np.array([])
    eval_psnrs = np.array([])

    for epoch, values in eval_stats.items():
        it = (epoch+1)*max_it
        psnr = values[0]

        eval_its = np.block([eval_its, it])
        eval_psnrs = np.block([eval_psnrs, psnr])

    plt.figure()
    plt.plot(train_its, train_psnrs, label = "Train PSNR")
    plt.plot(eval_its, eval_psnrs, label = "Validation PSNR")

    plt.ylabel('PSNR')
    plt.xlabel('Iterations')
    plt.title(title)

    plt.legend()
    plt.show()

"""
input_dict: {"label_0": "path_to_log_0", "label_1": "path_to_log_1", ...}
"""
def plot_evals_together(input_dict, title = ""):

    plt.figure()

    plt.ylabel('PSNR')
    plt.xlabel('Iterations')
    plt.title(title)

    for label, filepath in input_dict.items():
        eval_stats,train_stats = get_stats(filepath)
        max_it = 0

        train_its = np.array([])
        train_psnrs = np.array([])
        eval_its = np.array([])
        eval_psnrs = np.array([])

        for epoch, values in train_stats.items():
            train_epoch = np.array(values)
            it = train_epoch[:,0]
            psnr = train_epoch[:,1]

            if max_it == 0:
                max_it = it[-1]
                break

        for epoch, values in eval_stats.items():
            it = (epoch+1)*max_it
            psnr = values[0]

            eval_its = np.block([eval_its, it])
            eval_psnrs = np.block([eval_psnrs, psnr])
        
        #plt.plot(train_its, train_psnrs, label = "Train psnr")
        plt.plot(eval_its, eval_psnrs, label = label)

    plt.legend()
    plt.show()



log_folder = r"C:\Users\einar\OneDrive - NTNU\Semester 10\master project\logs"
input_dict = {"Wrong transformation": log_folder+os.sep+"lighthouse_wrong_ds", 
             "Correct transformation": log_folder+os.sep+"Lighthouse_colmapc2w"}

input_dict = {"Basic": log_folder+os.sep+"colmapc2w_good_example", 
             "Nearclip": log_folder+os.sep+"fruit_colmapc2w_nearclip",
             "Nearclip + Weight Decay": log_folder+os.sep+"fruit_fix",}
#plot_evals_together(input_dict, "Fruit Test PSNR")




eval_stats, train_stats = get_stats(log_folder+os.sep+"fruit_expanded_filtering_fixed_1103")
make_plots("Expanded ROI", eval_stats, train_stats)

eval_max, train_max = peak_psnr(eval_stats, train_stats)
print("training max:", train_max, "eval max:", eval_max)
