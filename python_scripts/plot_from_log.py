import json
import numpy as np
import matplotlib.pyplot as plt

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

#eval_stats: {-1 : [psnr,mse] , 0 : [psnr,mse], 1 : [psnr,mse], ....}
#train_stats: {0: [[it_0, psnr_0], [it_1, psnr_1], ...], 1:...}
def make_plots(eval_stats, train_stats):
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
    
    epoch_its = np.array([])
    epoch_psnrs = np.array([])

    for epoch, values in eval_stats.items():
        it = (epoch+1)*max_it
        psnr = values[0]

        epoch_its = np.block([epoch_its, it])
        epoch_psnrs = np.block([epoch_psnrs, psnr])

    plt.figure()
    plt.plot(train_its, train_psnrs, label = "Train psnr")
    plt.plot(epoch_its, epoch_psnrs, label = "Test psnr")

    plt.ylabel('Psnr')
    plt.xlabel('Iterations')
    plt.title('Colmap C2W transform')

    plt.xticks(np.arange(0, max_it*(epochs+1), step=max_it))
    plt.legend()
    plt.show()

eval_stats,train_stats = get_stats(r"C:\Users\einarjso\neodroid_plenoxels\python_scripts\log")
make_plots(eval_stats, train_stats)