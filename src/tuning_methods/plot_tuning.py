# python stuff
import sys
import pandas
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from matplotlib import pyplot as plt

# Load one configuration file here
if sys.argv[1] == 'vgg':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')# Load one configuration file here

if __name__ == "__main__":
    for peep_layer in target_layers:
        hyper_params_file = phs_path/f'hyperparams.{peep_layer}.pickle'
        results_df = pandas.read_pickle(hyper_params_file)
        data = results_df[['config/peep_size', 'config/n_classifier', 'topk_acc']].to_numpy()
        x = torch.tensor(data[:,0:2])
        y = torch.tensor(data[:, 2])
        print(f'\n {peep_layer} -----------\ntopk_acc: min = {y.min()} - max = {y.max()} - inproves: ', y.max()-y.min())
        
        best_config = x[y.argmax()]
        print(f'best config: peep_size = {best_config[0]}, n_classifier = {best_config[1]}, topk_acc={y.max()}')
        gp = GP()
        gp.fit(x, y)
        
        nps = max_peep_size[peep_layer]-50+1
        nnc = max_n_classifier-50+1
        ps, nc = torch.meshgrid(
                torch.linspace(50, max_peep_size[peep_layer], nps),
                torch.linspace(50, max_n_classifier, nnc),
                indexing = 'ij'
                )

        px = torch.vstack((ps.flatten(), nc.flatten())).T
        pred = gp.predict(px).reshape(nps, nnc)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(ps, nc, pred)
        ax.set_title(peep_layer)
        plt.savefig(f'plots/topk_acc_predictions.{peep_layer}.png')
