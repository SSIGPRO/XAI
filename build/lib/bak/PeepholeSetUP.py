#!/usr/bin/env python
# coding: utf-8

# # Results evaluation
# 

# ### UTILS


from utils_membership_prob import *



import cmasher as cmr
cmap = cmr.get_sub_cmap('brg', 0, 1)


import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'widget')


# ## Select device

use_cuda = torch.cuda.is_available()
cuda_index = torch.cuda.device_count() - 2
device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
print(f"Using {device} device")


# ## Configuration

dataset = 'CIFAR100'

fine_tuned = True

seed = 29

bs = 64

pretrained = True

model_name = 'vgg16'

dataset_name = dataset

len_train = 40000
len_val = 10000


#dims_list = [10, 15, 20, 25, 30]
dims_list = [10]
dims_list = [40, 50, 60, 70, 100]
# dims_list = [100]

num_clusters = [10, 15, 20, 50, 120, 150]

# num_clusters = [
#                 # 10, 15, 20, 50, 100, 
#                 120, 150,
#                 # 200, 250, 300,
#                 350, 500
# ]

method = 'KM'

layers_dict = {'feat': [24,26,28],
               'clas':[0,3],}

dir = 'in'

layer_list =['feat-24', 'feat-26', 'feat-28', 'clas-0', 'clas-3']


ATK = 'CW'


# ## Load

# ### Load data

batch_size = 64
data_kwargs = {'num_workers': 4, 'pin_memory': True}


# Get classes and `num_classes`


classes = load_data(dataset, 'classes', batch_size=batch_size, data_kwargs={})
num_classes = len(classes.keys())


train_loader = load_data(dataset, 
                         'train', 
                         batch_size=batch_size, 
                         data_kwargs=data_kwargs
                        )


val_loader = load_data(dataset, 
                       'val', 
                       batch_size=batch_size, 
                       data_kwargs=data_kwargs
                      )


test_loader = load_data(dataset, 
                       'test', 
                       batch_size=batch_size, 
                       data_kwargs=data_kwargs
                      )


# ### Load checkpoint
# Load checkpoint from a pretrained and fine-tuned model


model_name = f'vgg16_pretrained={pretrained}_dataset={dataset}-'\
             f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'



abs_models_dir = os.path.join('/srv/newpenny/XAI/LM', 'models')
chkp_path = os.path.join(abs_models_dir, model_name)

if os.path.exists(chkp_path):
    chkp = torch.load(chkp_path, map_location='cpu')
    state_dict = chkp['state_dict']



# see what is saved in the checkpoint (except for the state_dict)
for k, v in chkp.items():
    if k != 'state_dict':
        print(k, v)


# ### Load Activation


dict_activations_train = load_activations(activations_path, portion='train', dataset=dataset_name)



dict_activations_val = load_activations(activations_path, portion='val', dataset=dataset_name)


name = f'dict_activations_attack={ATK}-dataset={dataset_name}.pkl'
dict_activations_attack = load_res_c(name)


# ### Load SVD


## configuration

layers_dict = {'clas': [0,3],
              'feat':[24,26,28],
              }

fine_tuned = True

if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
    seed = 29
else:
    seed = 'nd'

layers = ''

for key in layers_dict.keys():
    for index in layers_dict[key]:
        layer = key + '_' + str(index) + '&'
        layers += layer

dict_file = f'dict_SVD_model={model_name}_layer={layers}_'\
               f'ft={fine_tuned}_seed={seed}_dataset={dataset_name}_.pkl'      

dict_file



path = os.path.join('data', 'SVD', dict_file)
with open(path, mode='rb') as fp:
    dict_SVD = pickle.load(fp)



dict_SVD.keys()


# ### Load model


model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)


# Modify output shape depending on `num_classes`


in_features = 4096
num_classes = 100
out_features = num_classes 
model.classifier[-1] = nn.Linear(in_features, out_features)



model.load_state_dict(state_dict)



model = model.to(device)
model.eval()


# ### Load attack dataset


path_ = os.path.join('data', 'attack')
with open(os.path.join(path_, f'adv_dict_train_method={ATK}.pkl'), 'rb') as fp:
    adv_dict = pickle.load(fp)



dataset_ = DictDataset(adv_dict)
 
attack_loader = torch.utils.data.DataLoader(dataset_, 
                                            batch_size=64, 
                                            shuffle=False)



dict_activations_train = load_activations(activations_path, portion='train', dataset=dataset_name)



dict_activations_val = load_activations(activations_path, portion='val', dataset=dataset_name)


# ### Load Cycles with EP

# #### Train & Val


dict_peephole_train_ = []
dict_peephole_val_ = []
empirical_posterior_ = []
distances_prob_train_ = []
distances_prob_val_ = []
clustering_labels_train_ = []
clustering_labels_val_ = []

for dim in dims_list:
    
    name = f'_dict_peephole_train-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    dict_peephole_train_.append(load_res_lc(name))
    name = f'_dict_peephole_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    dict_peephole_val_.append(load_res_lc(name))
    name = f'_empirical_posterior-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    empirical_posterior_.append(load_res_lc(name))
    name = f'_distances_prob_train-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    distances_prob_train_.append(load_res_lc(name))
    name = f'_distances_prob_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    distances_prob_val_.append(load_res_lc(name))
    name = f'_clustering_labels_train-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    clustering_labels_train_.append(load_res_lc(name))
    name = f'_clustering_labels_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    clustering_labels_val_.append(load_res_lc(name))



dict_peephole_train = {k: v for d in dict_peephole_train_ for k, v in d.items()}
dict_peephole_val = {k: v for d in dict_peephole_val_ for k, v in d.items()}
empirical_posterior = {k: v for d in empirical_posterior_ for k, v in d.items()}
distances_prob_train = {k: v for d in distances_prob_train_ for k, v in d.items()}
distances_prob_val = {k: v for d in distances_prob_val_ for k, v in d.items()}
clustering_labels_train = {k: v for d in clustering_labels_train_ for k, v in d.items()}
clustering_labels_val = {k: v for d in clustering_labels_val_ for k, v in d.items()}


# #### Attack


dict_peephole_attack_ = []

distances_prob_attack_ = []

clustering_labels_attack_ = []

for dim in dims_list:

    name = f'_dict_peephole_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    dict_peephole_attack_.append(load_res(name))

    name = f'_distances_prob_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    distances_prob_attack_.append(load_res(name))
    
    name = f'_clustering_labels_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    clustering_labels_attack_.append(load_res(name))



dict_peephole_attack = {k: v for d in dict_peephole_attack_ for k, v in d.items()}
distances_prob_attack = {k: v for d in distances_prob_attack_ for k, v in d.items()} 
clustering_labels_attack = {k: v for d in clustering_labels_attack_ for k, v in d.items()}


# ### Load Cycles WITHOUT EP


dist_attack_ = []

dist_val_ = []

for dim in dims_list:

    name = f'_distances_attack={ATK}-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    dist_attack_.append(load_res_lm(name))
    
    name = f'_distances_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    dist_val_.append(load_res_lm(name))



dist_attack = {k: v for d in dist_attack_ for k, v in d.items()} 
dist_val = {k: v for d in dist_val_ for k, v in d.items()}



dict_dist_val = {}
dict_dist_attack = {}

for dim in dims_list:

    for n in num_clusters:
        
        dict_dist_val[(dim,n)] = { 'feat-24' : dist_val[(dim,n)][-5],
                                   'feat-26' : dist_val[(dim,n)][-4],
                                   'feat-28' : dist_val[(dim,n)][-3],
                                   'clas-0' : dist_val[(dim,n)][-2],
                                   'clas-3' : dist_val[(dim,n)][-1],}
        
        dict_dist_attack[(dim,n)] = { 'feat-24' : dist_attack[(dim,n)][-5],
                                      'feat-26' : dist_attack[(dim,n)][-4],
                                      'feat-28' : dist_attack[(dim,n)][-3],
                                      'clas-0' : dist_attack[(dim,n)][-2],
                                      'clas-3' : dist_attack[(dim,n)][-1],}


# ## STORE

# ### Store SVD


## configuration

layers_dict = {'clas': [0,3],
              'feat':[28],}

model_name = 'vgg16'

dataset_name = 'CIFAR100'

fine_tuned = True

if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
    seed = 29
else:
    seed = 'nd'

layers = ''

for key in layers_dict.keys():
    for index in layers_dict[key]:
        layer = key + '_' + str(index) + '&'
        layers += layer

dict_file = f'dict_SVD_model={model_name}_layer={layers}_'\
               f'ft={fine_tuned}_seed={seed}_dataset={dataset_name}_.pkl'      

dict_file

path = os.path.join('data', 'SVD', dict_file)
with open(path, 'wb') as fp:
    pickle.dump(dict_SVD, fp)


# ### Store peephole


## configuration

layers_dict = {'clas': [0,3],
              'feat': [28]}

model_name = 'vgg16'

dataset_name = 'CIFAR100'

fine_tuned = True

if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
    seed = 29
else:
    seed = 'nd'

layers = ''

for key in layers_dict.keys():
    for index in layers_dict[key]:
        layer = key + '_' + str(index) + '&'
        layers += layer

portion = 'val'

dim = '50'

clustering = 'KM'

dict_file = f'dict_peephole_dim={dim}_portion={portion}_model={model_name}_layer={layers}_'\
               f'-clustering_{clustering}_ft={fine_tuned}_seed={seed}_dataset={dataset_name}.pkl'      

dict_file

path = os.path.join('data', 'peephole', dict_file)
with open(path, 'wb') as fp:
    pickle.dump(dict_peephole_val_50, fp)


# ### Store Empirical posterior


## configuration

if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
    seed = 29
else:
    seed = 'nd'

layers = ''

for key in layers_dict.keys():
    for index in layers_dict[key]:
        layer = key + '_' + str(index) + '&'
        layers += layer

EP_filename = f'empirical_posterior_model={model_name}_layer={layers}_'\
               f'ft={fine_tuned}_dim={dim}_seed={seed}_dataset={dataset_name}_.pkl'      

EP_filename

path = os.path.join('data', 'empirical_posterior', EP_filename)
with open(path, 'wb') as fp:
    pickle.dump(empirical_posterior_50, fp)


# ## Computation

# ### SVD


dict_SVD = {}

layer_type = 'classifier'

layer_index = '3'

layer_name = layer_type[0:4] + '-' + layer_index

U, s, Vh = get_svd(params=state_dict, layer_type=layer_type, layer_index=layer_index, input_shape=None, k=None)

dict_SVD[layer_name] = [U, s, Vh]


# #### Working with Conv


name_ = f'unrolled_params_layer=features-28_dataset=CIFAR100.npz'
abs_path = '/srv/newpenny/XAI/LM'

path_ = os.path.join(abs_path, 'data', 'peepholes', name_)
path_

W_csr_ = scipy.sparse.load_npz(path_)

U, s, Vh = get_svd_sparse(W_csr_, k=100)


# ### Activations

# #### Train

dict_activations_train = get_activation_VGG(model, 
                                            loader=train_loader, 
                                            layers_dict=layers_dict, 
                                            dir=dir, 
                                            device=device)


# #### Val


dict_activations_val = get_activation_VGG(model, 
                                          loader=val_loader, 
                                          layers_dict=layers_dict, 
                                          dir=dir, 
                                          device=device)


# #### Attack

dict_activations_attack = get_activation_VGG(model, 
                                             loader=attack_loader, 
                                             layers_dict=layers_dict, 
                                             dir=dir, 
                                             device=device)



## configuration

if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
    seed = 29
else:
    seed = 'nd'

layers = ''
model_name= 'vgg16'

for key in layers_dict.keys():
    for index in layers_dict[key]:
        layer = key + '_' + str(index) + '&'
        layers += layer
activations_attack = f'dict_activations_portion=train_attack_{ATK}_model={model_name}_layer={layers}_'\
                       f'dir={dir}_ft={fine_tuned}_seed={seed}_dataset={dataset_name}_.pkl'      
activations_attack

path_ = os.path.join('data', 'dict_activations', activations_attack)
with open(os.path.join(path_), 'wb') as fp:
    pickle.dump(dict_activations_attack,fp)


# ### Computation Cycle with EP

# #### Train & Val


dict_peephole_train = {} 
dict_peephole_val = {} 
empirical_posterior = {} 
distances_prob_train = {}
distances_prob_val = {}
clustering_labels_train = {}
clustering_labels_val = {}  

for dim in tqdm(dims_list):
    
    # 1. get peepholes train 
    n_ = 1
    n_clusters_ref = {}
    
    for key in layer_list:
        n_clusters_ref[key] = n_
    
    dict_peephole_train_ref = get_dict_peephole_train(dim=dim,
                                                      dict_activations_train=dict_activations_train,
                                                      n_clusters=n_clusters_ref,
                                                      dict_SVD=dict_SVD)
    
    # 2. get peepholes val
    dict_peephole_val_ref = get_dict_peephole_val(dim=dim, 
                                                  dict_activations_val=dict_activations_val, 
                                                  dict_peephole_train=dict_peephole_train_ref,
                                                  n_clusters=n_clusters_ref,
                                                  dict_SVD=dict_SVD)
    
    for n in num_clusters:
        
        n_clusters = {}

        for key in layer_list:
            n_clusters[key] = n
        
        # 3. get updated peepholes train 
        dict_peephole_train[(dim, n)] = get_clustering_config(dict_peephole_train=dict_peephole_train_ref,
                                                              n_clusters=n_clusters)

        # 4. get updated peepholes val
        dict_peephole_val[(dim, n)] = {'peephole': dict_peephole_val_ref['peephole'],
                                      'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        
        # 5. get empirical posteriors
        empirical_posterior[(dim, n)] = fit_empirical_posteriors(dict_activations_train=dict_activations_train, 
                                                                 dict_peephole_train=dict_peephole_train[(dim, n)], 
                                                                 n_classes=num_classes)
        
        # 6. get membership probability for training set
        distances_prob_train[(dim, n)] = get_distances_prob(dict_peephole_val=None,
                                                            dict_peephole_train=dict_peephole_train[(dim, n)],
                                                            method=method, 
                                                            dim=dim)
        
        # 7. get membership probability for training set
        distances_prob_val[(dim, n)] = get_distances_prob(dict_peephole_val=dict_peephole_val[(dim, n)],
                                                          dict_peephole_train=dict_peephole_train[(dim, n)],
                                                          method=method, 
                                                          dim=dim)
        
        # 8. combine clustering labels train
        clustering_labels_t = []
        distances_prob = distances_prob_train[(dim, n)]
        for element in zip(*distances_prob):
            ll = element
            clustering_labels_t.append(ll)
        clustering_labels_train[(dim, n)] = clustering_labels_t
            
        #9. combine clustering labels val
        clustering_labels_v = []
        distances_prob = distances_prob_val[(dim, n)]
        for element in zip(*distances_prob):
            ll = element
            clustering_labels_v.append(ll)
        clustering_labels_val[(dim, n)] = clustering_labels_v
        
        
    name = f'_dict_peephole_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, dict_peephole_train)
    name = f'_dict_peephole_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, dict_peephole_val)
    name = f'_empirical_posterior-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, empirical_posterior)
    name = f'_distances_prob_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, distances_prob_train)
    name = f'_distances_prob_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, distances_prob_val)
    name = f'_clustering_labels_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, clustering_labels_train)
    name = f'_clustering_labels_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res_lc(name, clustering_labels_val)
    


# #### Attack


dict_peephole_attack = {}
distances_prob_attack = {} 
clustering_labels_attack = {} 

for dim in dims_list:
    
    # 1. get peepholes train 
    n_ = 10
    n_clusetrs_ref = {}
    
    for key in layer_list:
        n_clusters_ref[key] = n_
    
    dict_peephole_train_ref = dict_peephole_train[(dim, n_)]
    
    dict_peephole_attack_ref = get_dict_peephole_val(dim=dim, 
                                                     dict_activations_val=dict_activations_attack, 
                                                     dict_peephole_train=dict_peephole_train_ref, # vedere
                                                     n_clusters=n_clusters_ref,
                                                     dict_SVD=dict_SVD)
    
    for n in num_clusters:
        
        dict_peephole_attack[(dim, n)] = {'peephole': dict_peephole_attack_ref['peephole'],
                                          'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        
        distances_prob_attack[(dim, n)] = get_distances_prob(dict_peephole_val=dict_peephole_attack[(dim, n)],
                                                             dict_peephole_train=dict_peephole_train[(dim, n)],
                                                             method=method, 
                                                             dim=dim)
        
        clustering_labels_a = []
        distances_prob = distances_prob_attack[(dim, n)]
        for element in zip(*distances_prob):
            ll = element
            clustering_labels_a.append(ll)
        clustering_labels_attack[(dim, n)] = clustering_labels_a
    
    name = f'_dict_peephole_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, dict_peephole_attack)
    name = f'_distances_prob_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, distances_prob_attack)
    name = f'_clustering_labels_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, clustering_labels_attack)


# ### Computation cycle WITHOUT EP


dict_peephole_attack = {}
dist_attack = {}  

for dim in tqdm(dims_list):
    
    # 1. get peepholes train 
    n_ = 10
    n_clusetrs_ref = {}
    
    for key in layer_list:
        n_clusters_ref[key] = n_
    
    dict_peephole_train_ref = dict_peephole_train[(dim, n_)]
    
    dict_peephole_attack_ref = get_dict_peephole_val(dim=dim, 
                                                     dict_activations_val=dict_activations_attack, 
                                                     dict_peephole_train=dict_peephole_train_ref, 
                                                     n_clusters=n_clusters_ref,
                                                     dict_SVD=dict_SVD)
    
    dict_peephole_val_ref = get_dict_peephole_val(dim=dim, 
                                                  dict_activations_val=dict_activations_val, 
                                                  dict_peephole_train=dict_peephole_train_ref, 
                                                  n_clusters=n_clusters_ref,
                                                  dict_SVD=dict_SVD)
    
    for n in tqdm(num_clusters):
        
            
        dict_peephole_attack[(dim, n)] = {'peephole': dict_peephole_attack_ref['peephole'],
                                          'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        dist_attack_ = get_distances(dict_peephole_val=dict_peephole_attack[(dim,n)], 
                                     dict_peephole_train=dict_peephole_train[(dim,n)], 
                                     method=method, dim=dim)
        dist_attack[(dim,n)] = dist_attack_

        dict_peephole_val[(dim, n)] = {'peephole': dict_peephole_val_ref['peephole'],
                                       'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        dist_val_ = get_distances(dict_peephole_val=dict_peephole_val[(dim,n)], 
                                  dict_peephole_train=dict_peephole_train[(dim,n)], 
                                  method=method, dim=dim)
        dist_val[(dim,n)] = dist_val_
    
    name = f'_dict_peephole_attack={ATK}-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    save_res(name, dict_peephole_attack)
    name = f'_distances_attack={ATK}-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    save_res(name, dist_attack)
    name = f'_dict_peephole_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    save_res(name, dict_peephole_val)
    name = f'_distances_val-dim={dim}-method={method}-dataset={dataset_name}.pkl'
    save_res(name, dist_val)


# ## Analysis of the attacks


# Define the CIFAR-100 classes
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

# Create a dictionary where each key is the numeric class and the value is the class name
class_dict = {i: cifar100_classes[i] for i in range(len(cifar100_classes))}

print(class_dict)


# ## DEPLETION


results = np.concatenate(dict_activations_val['results'])
tot_true = np.sum(results==True)
tot_false = np.sum(results==False)
tot_true, tot_false


# ### Empirical Posterior = True


def predict_proba_v(weights=None, empirical_posterior=None, clustering_labels=None, n_classes=None):
    """Returns surrogate model's predicted probabilities.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      weights: a list of weights (need not be normalized), one for each
        activation.
    """
    # Equal weights are used if not provided by the user.
    if weights is None:
        weights = [1.0] * len(activation_names)

    pred = []
    for one_hot_encoding in clustering_labels:
        # print(one_hot_encoding)
        # print(len(one_hot_encoding))
        pred.append(p(example_score=one_hot_encoding[0], 
                      empirical_posterior=empirical_posterior, 
                      weights=weights, n_classes=n_classes))
    return np.array(pred)


measure = 'max'

weights_24 = [1, 0, 0, 0, 0]
weights_26 = [0, 1, 0, 0, 0]
weights_28 = [0, 0, 1, 0, 0]
weights_0 = [0, 0, 0, 1, 0]
weights_3 = [0, 0, 0, 0, 1]
weights_unbalanced_1 = [0, 0, 0, 0.2, 0.8]
weights_unbalanced_2 = [0, 0, 0, 0.1, 0.9]
weights_unbalanced_3 = [0.33,0.33,0.33,0,0]


w_dict = {
          # 'feat-24' : weights_24,
          # 'feat-26' : weights_26,
          # 'feat-28' : weights_28,
          'clas-0' : weights_0,
          'clas-3' : weights_3, 
          '0.2-0.8' : weights_unbalanced_1,
          '0.1-0.9' : weights_unbalanced_2,
          # 'features' : weights_unbalanced_3, 
          }

dims_list, num_clusters

len_dict = len(w_dict.keys())
len_dim = len(dims_list)
array = np.arange(0,100,0.1)


# ##### Fine


fig, axs = plt.subplots(len_dict,len_dim,figsize=(20,16))

for j, (key, weight) in enumerate(w_dict.items()):

    for k, dim in enumerate(dims_list):
        if j==0:
            axs[j,k].set_title(dim)
        
        axs[j,k].plot([0, 100],[1,0],label='ref', c='k', ls='--')
        
        for n in num_clusters:

            ep = empirical_posterior[(dim,n)]
            cv = clustering_labels_val[(dim,n)]
            ct = clustering_labels_train[(dim,n)]

            prob_train = predict_proba_t(weights=weight, 
                                         empirical_posterior=ep, 
                                         clustering_labels=ct, 
                                         n_classes=100)
            
            prob_val = predict_proba_v(weights=weight, 
                                       empirical_posterior=ep, 
                                       clustering_labels=cv, 
                                       n_classes=100)
            
            conf_t = np.max(prob_train,axis=1)
            conf_v = np.max(prob_val,axis=1)
            
            threshold = []
            list_true_max_ = []
            list_false_max_ = []
            
            for i in array:
            
                perc = np.percentile(conf_t, i)
                
                threshold.append(perc)
                idx = np.argwhere(conf_v>perc)[:,0]
                counter = collections.Counter(results[idx])
                list_true_max_.append(counter[True]/tot_true)
                list_false_max_.append(counter[False]/tot_false)  

            if k==0:
                axs[j,k].set_ylabel(key)

            axs[j,k].plot(array,list_true_max_)
            axs[j,k].plot(array,list_false_max_,label=f'{n}')
            axs[j,k].grid()
            
            # axs[j].set_title(f'weights={formatted_weight}')
            #axs[j,k].title(f'dim={dim} num_clusters={n}', fontsize=16)
            # axs[j].legend()
            
fig.tight_layout()
fig.subplots_adjust(top=0.9)

def save_res(name, file):
    path = os.path.join(abs_lm,'results', 'paper', name)
    with open(path, 'wb') as fp:
        pickle.dump(file, fp)

for key, weight in w_dict.items():

    for n in num_clusters: 
        
        fig, axs = plt.subplots(1,figsize=(10,10))
        axs.grid()
        axs.plot([0, 100],[1,0],label='ref', c='k', ls='--')
        cmap = plt.get_cmap('Blues')
        plt.set_cmap(cmap)
        
        for dim in dims_list: 
            
            ep = empirical_posterior[(dim,n)]
            cv = clustering_labels_val[(dim,n)]
            ct = clustering_labels_train[(dim,n)]

            prob_train = predict_proba_t(weights=weight, 
                                         empirical_posterior=ep, 
                                         clustering_labels=ct, 
                                         n_classes=100)
            
            if dim == 10 or dim== 15 or dim == 20 or dim == 25 or dim == 30:
                

                if n == 10 or n == 15 or n == 20 or n == 50 or n == 120 or n == 150:
                
                    prob_val = predict_proba_v(weights=weight, 
                                               empirical_posterior=ep, 
                                               clustering_labels=cv, 
                                               n_classes=100)
    
                else:
                    prob_val = predict_proba_t(weights=weight, 
                                               empirical_posterior=ep, 
                                               clustering_labels=cv, 
                                               n_classes=100)
            else:
                prob_val = predict_proba_t(weights=weight, 
                                               empirical_posterior=ep, 
                                               clustering_labels=cv, 
                                               n_classes=100)
            
            conf_t = np.max(prob_train,axis=1)
            conf_v = np.max(prob_val,axis=1)
            
            threshold = []
            list_true_max_ = []
            list_false_max_ = []
            
            for i in array:
            
                perc = np.percentile(conf_t, i)
                
                threshold.append(perc)
                idx = np.argwhere(conf_v>perc)[:,0]
                counter = collections.Counter(results[idx])
                list_true_max_.append(counter[True]/tot_true)
                list_false_max_.append(counter[False]/tot_false)  

            axs.plot(array,list_true_max_)
            axs.plot(array,list_false_max_,label=f'{dim}')
            
            axs.legend()
            fig.suptitle(f'RF of MAX num clusters={n} layer={key}', fontsize=16)
            
fig.tight_layout()
fig.subplots_adjust(top=0.9)


# #### Attack visualization


data_iterator = iter(val_loader)
attack_iterator = iter(attack_loader)

n = 18

for i in range(n):
    data, label = next(data_iterator)
    attack, target = next(attack_iterator)

data, label = data.to(device), label.to(device)
attack, target = attack.to(device), target.to(device)

model = model.to(device)
model.eval()

out = model(data)
out_attack = model(attack)

sm = nn.Softmax()

prob = sm(out)
prob_attack = sm(out_attack)

final_prob = prob.argmax(1, keepdim=True).detach().cpu().numpy()
final_prob_attack = prob_attack.argmax(1, keepdim=True).detach().cpu().numpy()

fig, axs = plt.subplots(1,3,figsize=(15,8))
k=8

# Plot data on each subplot
axs[0].imshow(np.transpose(data[k].squeeze().detach().cpu().numpy(), (1, 2, 0)))

# Add titles to each subplot
axs[0].set_title(class_dict[int(final_prob[k])])

# Plot data on each subplot
axs[1].imshow(np.transpose(attack[k].squeeze().detach().cpu().numpy(), (1, 2, 0)))

# Add titles to each subplot
axs[1].set_title(class_dict[int(final_prob_attack[k])])
# Plot data on each subplot
axs[2].imshow(np.transpose((data[k]-attack[k]).squeeze().detach().cpu().numpy(), (1, 2, 0)))

print(final_prob[k])
print(torch.max(prob[k]))
print(label[k])
print(final_prob_attack[k])
print(torch.max(prob_attack[k]))
print(target[k])

model.eval()
sm = nn.Softmax()


entropy_val = []
number_corrects = 0
number_samples = 0    
  

for data in tqdm(val_loader):
    image,label = data
    
    image = image.to(device)
    label = label.to(device)
    
    
    out = model(image)    
    prob = sm(out)

    final_prob = prob.argmax(axis=1)
    number_corrects += (label==final_prob).sum().item()
    number_samples += label.size(0)    
    
    entropy = H(prob.detach().cpu().numpy(),axis=1)
    entropy_val.append(entropy)

print(f'Overall accuracy {(number_corrects / number_samples)*100}%')
accuracy_nearest= (number_corrects / number_samples)*100

entropy_attack = []
number_corrects = 0
number_samples = 0

for data in tqdm(attack_loader):
    image,label = data
    
    image = image.to(device)
    label = label.to(device)
    
    out = model(image)    
    prob = sm(out)

    final_prob = prob.argmax(axis=1)
    number_corrects += (label==final_prob).sum().item()
    number_samples += label.size(0)  
    
    entropy = H(prob.detach().cpu().numpy(),axis=1)
    entropy_attack.append(entropy)

print(f'Attack success rate {(number_corrects / number_samples)*100}%')
attack_success = (number_corrects / number_samples)*100

H_val = np.concatenate(entropy_val)

H_attack = np.concatenate(entropy_attack)

fig, axs = plt.subplots(1,figsize=(10,8))
axs.hist(H_val, bins=50, density=True, alpha=0.5, label='val')
axs.hist(H_attack, bins=30, density=True, alpha=0.5, label='attack')
axs.set_title('Entropy')
axs.legend()

plt.show()

original_labels = np.zeros(len(H_val))
attack_labels = np.ones(len(H_attack))

y_true = np.concatenate([original_labels, attack_labels])

y_score_NN = np.concatenate([H_val,H_attack])

auc_NN = AUC(y_true=y_true, y_score=y_score_NN)

auc_NN


# ### Analysis of the attacks on peephole

# #### ENTROPY without SOFTMIN

# ##### Monitoring layers separately

# ###### no likelihood


for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = H(dv, axis=1)
            score_a = H(da, axis=1)

            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([score_v, score_a])

            auc_ = AUC(y_true=y_true, y_score=y_score)
            
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ###### with likelihood

for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = H(dv, axis=1)
            score_a = H(da, axis=1)
            
            kde_scott = KDE(score_v, bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(score_v))
            lh_attack = -np.log(kde_scott.pdf(score_a))
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ##### Monitoring the Joint PDF

dict_matrix_attack = {}
dict_matrix_val = {}

for n in num_clusters:

    for dim in dims_list:
        
        matrix_val = np.zeros((len_val,len(layer_list)))
        matrix_attack = np.zeros((len_val,len(layer_list)))

        for i,layer in enumerate(layer_list):
        
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = H(dv, axis=1)
            score_a = H(da, axis=1)  
            
            matrix_attack[:,i] = score_a
            matrix_val[:,i] = score_v
            
        dict_matrix_attack[(dim,n)] = matrix_attack
        dict_matrix_val[(dim,n)] = matrix_val
                        

list_layers = ['clas3', 'clas0-3', 'feat28+clas0-3', 'feat26-28+clas0-3', 'feat24-26-28+clas0-3']

for i in range(1,6):
    
    data = []
    
    for n in num_clusters:
    
        for dim in dims_list:
    
            kde_scott = KDE(dict_matrix_val[(dim,n)].T[-i:], bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(dict_matrix_val[(dim,n)].T[-i:]))
            lh_attack = -np.log(kde_scott.pdf(dict_matrix_attack[(dim,n)].T[-i:]))
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if i==1: # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)              

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# #### ENTROPY with SOFTMIN

# ##### Monitoring layers separately

# ###### without likelihood


for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            _val = torch.Tensor(dv)
            _attack = torch.Tensor(da)

            pv = torch.nn.functional.softmin(_val, dim=1)
            pa = torch.nn.functional.softmin(_attack, dim=1)

            score_v = H(pv.detach().cpu().numpy(), axis=1)
            score_a = H(pa.detach().cpu().numpy(), axis=1)

            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([score_v, score_a])

            auc_ = AUC(y_true=y_true, y_score=y_score)
            
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ###### with likelihood


for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            _val = torch.Tensor(dv)
            _attack = torch.Tensor(da)

            pv = torch.nn.functional.softmin(_val, dim=1)
            pa = torch.nn.functional.softmin(_attack, dim=1)

            score_v = H(pv.detach().cpu().numpy(), axis=1)
            score_a = H(pa.detach().cpu().numpy(), axis=1)
            
            kde_scott = KDE(score_v, bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(score_v))
            lh_attack = -np.log(kde_scott.pdf(score_a))
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ##### Monitoring the Joint PDF

dict_matrix_attack = {}
dict_matrix_val = {}

for n in num_clusters:

    for dim in dims_list:
        
        matrix_val = np.zeros((len_val,len(layer_list)))
        matrix_attack = np.zeros((len_val,len(layer_list)))

        for i,layer in enumerate(layer_list):
        
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            _val = torch.Tensor(dv)
            _attack = torch.Tensor(da)

            pv = torch.nn.functional.softmin(_val, dim=1)
            pa = torch.nn.functional.softmin(_attack, dim=1)

            score_v = H(pv.detach().cpu().numpy(), axis=1)
            score_a = H(pa.detach().cpu().numpy(), axis=1)    
            
            matrix_attack[:,i] = score_a
            matrix_val[:,i] = score_v
            
        dict_matrix_attack[(dim,n)] = matrix_attack
        dict_matrix_val[(dim,n)] = matrix_val
                        

list_layers = ['clas3', 'clas0-3', 'feat28+clas0-3', 'feat26-28+clas0-3', 'feat24-26-28+clas0-3']

for i in range(1,6):
    
    data = []
    
    for n in num_clusters:
    
        for dim in dims_list:
    
            kde_scott = KDE(dict_matrix_val[(dim,n)].T[-i:], bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(dict_matrix_val[(dim,n)].T[-i:]))
            lh_attack = -np.log(kde_scott.pdf(dict_matrix_attack[(dim,n)].T[-i:]))
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if i==1: # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)              

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# #### MIN

# ##### Monitoring layers separately

# ###### without likelihood

dict_min = {}
dict_attack = {}

for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = np.min(dv, axis=1)
            score_a = np.min(da, axis=1)

            dict_min[(n, dim)] = score_v
            dict_attack[(n, dim)] = score_a

    #         original_labels = np.zeros(len_val)
    #         attack_labels = np.ones(len_val)

    #         y_true = np.concatenate([original_labels, attack_labels])

    #         y_score = np.concatenate([score_v, score_a])

    #         auc_ = AUC(y_true=y_true, y_score=y_score)
            
    #         data.append((dim, n, auc_))

    # if layer=='feat-24': # starting building results dataframe
    #     df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
    #     df.set_index(['dim', 'n_clusters'], inplace=True)
    #     df.sort_index(inplace=True)
    # else: # concatenate with previous results
    #     df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
    #     df_.set_index(['dim', 'n_clusters'], inplace=True)
    #     df_.sort_index(inplace=True)
    #     df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)

for n in num_clusters:
    for dim in dims_list:
        fig, axs = plt.subplots(1,figsize=(10,8))
        axs.hist(dict_min[(n,dim)], bins=50, density=True, alpha=0.5, label='val')
        axs.hist(dict_attack[(n,dim)], bins=30, density=True, alpha=0.5, label='attack')
        axs.set_title('Entropy')
        axs.legend()
        
        plt.show()




# ###### with likelihood


for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = np.min(dv, axis=1)
            score_a = np.min(da, axis=1)
            
            kde_scott = KDE(score_v, bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(score_v))
            lh_attack = -np.log(kde_scott.pdf(score_a))
            lh_attack = lh_attack[np.isfinite(lh_attack)]
    
            original_labels = np.zeros(len(lh_val))
            attack_labels = np.ones(len(lh_attack))
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ##### Monitoring the Joint PDF


dict_matrix_attack = {}
dict_matrix_val = {}

for n in num_clusters:

    for dim in dims_list:
        
        matrix_val = np.zeros((len_val,len(layer_list)))
        matrix_attack = np.zeros((len_val,len(layer_list)))

        for i,layer in enumerate(layer_list):
        
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            score_v = np.min(dv, axis=1)
            score_a = np.min(da, axis=1)   
            
            matrix_attack[:,i] = score_a
            matrix_val[:,i] = score_v
            
        dict_matrix_attack[(dim,n)] = matrix_attack
        dict_matrix_val[(dim,n)] = matrix_val
                        

len(lh_attack[np.isfinite(lh_attack)])

list_layers = ['clas3', 'clas0-3', 'feat28+clas0-3', 'feat26-28+clas0-3', 'feat24-26-28+clas0-3']

for i in range(1,6):
    
    data = []
    
    for n in num_clusters:
    
        for dim in dims_list:
    
            kde_scott = KDE(dict_matrix_val[(dim,n)].T[-i:], bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(dict_matrix_val[(dim,n)].T[-i:]))
            lh_attack = -np.log(kde_scott.pdf(dict_matrix_attack[(dim,n)].T[-i:]))
            lh_attack = lh_attack[np.isfinite(lh_attack)]
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len(lh_attack))
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])            
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if i==1: # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)              



df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# #### MAX with SOFTMIN

# ##### Monitoring layers separately


for layer in layer_list:
    
    data = []
    for n in num_clusters:

        for dim in dims_list:
            
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            _val = torch.Tensor(dv)
            _attack = torch.Tensor(da)

            pv = torch.nn.functional.softmin(_val, dim=1)
            pa = torch.nn.functional.softmin(_attack, dim=1)

            score_v = np.max(pv.detach().cpu().numpy(), axis=1)
            score_a = np.max(pa.detach().cpu().numpy(), axis=1)

            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([score_v, score_a])

            auc_ = AUC(y_true=y_true, y_score=y_score)
            
            data.append((dim, n, auc_))

    if layer=='feat-24': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', layer])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)
                        


df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# ##### Monitoring the Joint PDF


dict_matrix_attack = {}
dict_matrix_val = {}

for n in num_clusters:

    for dim in dims_list:
        
        matrix_val = np.zeros((len_val,len(layer_list)))
        matrix_attack = np.zeros((len_val,len(layer_list)))

        for i,layer in enumerate(layer_list):
        
            dv = dict_dist_val[(dim,n)][layer]
            da = dict_dist_attack[(dim,n)][layer]

            _val = torch.Tensor(dv)
            _attack = torch.Tensor(da)

            pv = torch.nn.functional.softmin(_val, dim=1)
            pa = torch.nn.functional.softmin(_attack, dim=1)

            score_v = np.max(pv.detach().cpu().numpy(), axis=1)
            score_a = np.max(pa.detach().cpu().numpy(), axis=1)    
            
            matrix_attack[:,i] = score_a
            matrix_val[:,i] = score_v
            
        dict_matrix_attack[(dim,n)] = matrix_attack
        dict_matrix_val[(dim,n)] = matrix_val
                        

list_layers = ['clas3', 'clas0-3', 'feat28+clas0-3', 'feat26-28+clas0-3', 'feat24-26-28+clas0-3']

for i in range(1,6):
    
    data = []
    
    for n in num_clusters:
    
        for dim in dims_list:
    
            kde_scott = KDE(dict_matrix_val[(dim,n)].T[-i:], bw_method='scott')
            lh_val = -np.log(kde_scott.pdf(dict_matrix_val[(dim,n)].T[-i:]))
            lh_attack = -np.log(kde_scott.pdf(dict_matrix_attack[(dim,n)].T[-i:]))
    
            original_labels = np.zeros(len_val)
            attack_labels = np.ones(len_val)
    
            y_true = np.concatenate([original_labels, attack_labels])
    
            y_score = np.concatenate([lh_val, lh_attack])
    
            auc_ = AUC(y_true=y_true, y_score=y_score)
    
            data.append((dim, n, auc_))

    if i==1: # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', list_layers[i-1]])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)              



df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


# #### Entropy


#KDE_silver = {}
KDE_scott = {}

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
            
    data = []
    for dim in dims_list:
        for n in num_clusters:

            entropy_val = H(prob_val[(dim, n)],axis=1)
            entropy_attack = H(prob_attack[(dim, n)],axis=1)

            kde_scott = KDE(entropy_val, bw_method='scott')
            kde_silver = KDE(entropy_val, bw_method='silverman')
            KDE_scott[(dim,n)] = kde_scott
            #KDE_silver[(dim,n)] = kde_silver
            lh_val = -np.log(kde_scott.pdf(entropy_val))
            lh_attack = -np.log(kde_scott.pdf(entropy_attack))
            
            original_labels = np.zeros(len(entropy_val))
            attack_labels = np.ones(len(entropy_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([lh_val, lh_attack])

            auc_ = AUC(y_true=y_true, y_score=y_score)

            data.append((dim, n, auc_))
            

    if key=='feat-28': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)



df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)


dim = 15
n = 15

weights = [0, 1, 0]
ep = empirical_posterior[(dim, n)]
            
clv = clustering_labels_val_[(dim, n)]
cla = clustering_labels_attack[(dim, n)]

pv = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=clv, 
                   n_classes=num_classes)

pa = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=cla, 
                   n_classes=num_classes)

H_val = H(pv,axis=1)
H_attack = H(pa,axis=1)

kde_ = KDE(H_val, bw_method='scott')
#kde_silver = KDE(entropy_val, bw_method='silverman')
# kde_ = KDE_scott[(dim,n)]
#KDE_silver[(dim,n)] = kde_silver

#lh_val = -np.log(kde_.pdf(H_val))
#lh_attack = -np.log(kde_.pdf(H_attack))

eps_ = (np.max(H_val)-np.min(H_val))/1000
lh_val = 1- eps_ * kde_.pdf(H_val)
lh_attack = 1- eps_ * kde_.pdf(H_attack)

original_labels = np.zeros(len(H_val))
attack_labels = np.ones(len(H_attack))

y_true = np.concatenate([original_labels, attack_labels])

y_score = np.concatenate([lh_val, lh_attack])

auc_ = AUC(y_true=y_true, y_score=y_score)



auc_


lh_val


lh_attack


x_test = np.linspace(3.20,4.65,100)
y_test = kde_.pdf(x_test)
plt.figure()
plt.plot(x_test,y_test)


lh_val

lh_attack

fig, axs = plt.subplots(2,2,figsize=(10,8),sharey='col')
axs[0,0].hist(lh_val, bins=10, alpha=0.5, label='val')
axs[1,0].hist(lh_attack, bins=10, alpha=0.5, label='attack')
axs[0,1].hist(H_val, bins=10, alpha=0.5, label='val')
axs[1,1].hist(H_attack, bins=10, alpha=0.5, label='attack')
for ax in axs.ravel(): 
    ax.set(yscale='log')

# axs.set_title('Entropy')
# axs.legend()

plt.show()

weights_eq = [1/3, 1/3, 1/3]
# weights_24 = [1, 0, 0, 0, 0]
# weights_26 = [0, 1, 0, 0, 0]
weights_28 = [1, 0, 0]
weights_0 = [0, 1, 0]
weights_3 = [0, 0, 1]
# weights_28 = [1, 0, 0]
weights_0_3 = [ 0, 0.5, 0.5]
weights_28_0 = [0.5, 0.5, 0]

# w_list = [weights_eq, weights_3, weights_0, weights_28]

w_dict = {#'equal' : weights_eq, 
          'feat-28' : weights_28,
          'clas-3' : weights_3, 
          'clas-0' : weights_0, 
          'clas-0-3' : weights_0_3, 
          'feat-28-clas-0' : weights_28_0
         }

#KDE_silver = {}
KDE_scott = {}

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
            
    data = []
    data_1 = []
    for dim in dims_list:
        for n in num_clusters:

            entropy_val = H(prob_val[(dim, n)],axis=1)
            entropy_attack = H(prob_attack[(dim, n)],axis=1)

            kde_scott = KDE(entropy_val, bw_method='scott')
            kde_silver = KDE(entropy_val, bw_method='silverman')
            KDE_scott[(dim,n)] = kde_scott

            lh_val = -np.log(kde_scott.pdf(entropy_val))
            lh_attack = -np.log(kde_scott.pdf(entropy_attack))
            
            original_labels = np.zeros(len(entropy_val))
            attack_labels = np.ones(len(entropy_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([lh_val, lh_attack])
            auc_ = AUC(y_true=y_true, y_score=y_score)
            data.append((dim, n, auc_))
            

            y_score_1 = np.concatenate([entropy_val, entropy_attack])
            auc_1 = AUC(y_true=y_true, y_score=y_score_1)
            data_1.append((dim, n, auc_1))
            

    if key=='feat-28': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

    if key=='feat-28': # starting building results dataframe
        df1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df1.set_index(['dim', 'n_clusters'], inplace=True)
        df1.sort_index(inplace=True)
    else: # concatenate with previous results
        df_1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df_1.set_index(['dim', 'n_clusters'], inplace=True)
        df_1.sort_index(inplace=True)
        df1 = df1.join(df_1)

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)

df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1)

from IPython.display import display, HTML

def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

# Assuming that dataframes df1 and df2 are already defined
display_side_by_side(df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1), df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1))

#KDE_silver = {}
KDE_scott = {}

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
            
    data = []
    data_1 = []
    for dim in dims_list:
        for n in num_clusters:

            entropy_val = H(prob_val[(dim, n)],axis=1)
            entropy_attack = H(prob_attack[(dim, n)],axis=1)

            kde_scott = KDE(entropy_val, bw_method='scott')
            kde_silver = KDE(entropy_val, bw_method='silverman')
            KDE_scott[(dim,n)] = kde_scott
            eps_ = (np.max(H_val)-np.min(H_val))/1000
            lh_val = eps_ * kde_scott.pdf(H_val)
            lh_attack = eps_ * kde_scott.pdf(H_attack)
            
            original_labels = np.zeros(len(entropy_val))
            attack_labels = np.ones(len(entropy_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([lh_val, lh_attack])
            auc_ = AUC(y_true=y_true, y_score=y_score)
            
            data.append((dim, n, auc_))
            
            lh_val = 1 - eps_ * kde_scott.pdf(H_val)
            lh_attack = 1 - eps_ * kde_scott.pdf(H_attack)

            y_score_1 = np.concatenate([lh_val, lh_attack])
            auc_1 = AUC(y_true=y_true, y_score=y_score_1)
            
            data_1.append((dim, n, auc_1))
            

    if key=='feat-28': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

    if key=='feat-28': # starting building results dataframe
        df1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df1.set_index(['dim', 'n_clusters'], inplace=True)
        df1.sort_index(inplace=True)
    else: # concatenate with previous results
        df_1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df_1.set_index(['dim', 'n_clusters'], inplace=True)
        df_1.sort_index(inplace=True)
        df1 = df1.join(df_1)

# Assuming that dataframes df1 and df2 are already defined
display_side_by_side(df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1), df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1))


# #### MAX

#KDE_silver = {}
KDE_scott = {}

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
            
    data = []
    for dim in dims_list[:-1]:
        for n in num_clusters:

            max_val = np.max(prob_val[(dim, n)],axis=1)
            max_attack = np.max(prob_attack[(dim, n)],axis=1)

            kde_scott = KDE(max_val, bw_method='scott')
            kde_silver = KDE(max_val, bw_method='silverman')
            KDE_scott[(dim,n)] = kde_scott
            #KDE_silver[(dim,n)] = kde_silver
            lh_val = -np.log(kde_scott.pdf(max_val))
            lh_attack = -np.log(kde_scott.pdf(max_attack))
            
            original_labels = np.zeros(len(max_val))
            attack_labels = np.ones(len(max_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([lh_val, lh_attack])
            
            auc_ = AUC(y_true=y_true, y_score=y_score)

            data.append((dim, n, auc_))

    if key=='feat-28': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

y_score, max_val, max_attack

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)

#KDE_silver = {}
KDE_scott = {}

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
            
    data = []
    data_1 = []
    for dim in dims_list:
        for n in num_clusters:

            max_val = np.max(prob_val[(dim, n)],axis=1)
            max_attack = np.max(prob_attack[(dim, n)],axis=1)

            kde_scott = KDE(max_val, bw_method='scott')
            kde_silver = KDE(max_val, bw_method='silverman')
            KDE_scott[(dim,n)] = kde_scott
            #KDE_silver[(dim,n)] = kde_silver
            lh_val = -np.log(kde_scott.pdf(max_val))
            lh_attack = -np.log(kde_scott.pdf(max_attack))
            
            original_labels = np.zeros(len(max_val))
            attack_labels = np.ones(len(max_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([lh_val, lh_attack])
            
            auc_ = AUC(y_true=y_true, y_score=y_score)

            data.append((dim, n, auc_))

            y_score = np.concatenate([max_val, max_attack])

            auc_1 = AUC(y_true=y_true, y_score=y_score)

            data_1.append((dim, n, auc_1))

    if key=='feat-28': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

    if key=='feat-28': # starting building results dataframe
        df1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df1.set_index(['dim', 'n_clusters'], inplace=True)
        df1.sort_index(inplace=True)
    else: # concatenate with previous results
        df_1 = pd.DataFrame(data_1, columns=['dim', 'n_clusters', key])
        df_1.set_index(['dim', 'n_clusters'], inplace=True)
        df_1.sort_index(inplace=True)
        df1 = df1.join(df_1)

# Assuming that dataframes df1 and df2 are already defined
display_side_by_side(df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1), df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1))

dim = 10
n = 120

weights = [0, 0, 1]
ep = empirical_posterior[(dim, n)]
            
clv = clustering_labels_val_[(dim, n)]
cla = clustering_labels_attack[(dim, n)]

pv = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=clv, 
                   n_classes=num_classes)

pa = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=cla, 
                   n_classes=num_classes)

H_val = H(pv,axis=1)
H_attack = H(pa,axis=1)

kde_ = KDE(H_val, bw_method='scott')
#kde_silver = KDE(entropy_val, bw_method='silverman')
# kde_ = KDE_scott[(dim,n)]
#KDE_silver[(dim,n)] = kde_silver

lh_val = -np.log(kde_.pdf(H_val))
lh_attack = -np.log(kde_.pdf(H_attack))
#lh_val = kde_.pdf(H_val)
#lh_attack = kde_.pdf(H_attack)

original_labels = np.zeros(len(H_val))
attack_labels = np.ones(len(H_attack))

y_true = np.concatenate([original_labels, attack_labels])

y_score = np.concatenate([lh_val, lh_attack])

auc_ = AUC(y_true=y_true, y_score=y_score)

fig, axs = plt.subplots(2,figsize=(10,8),sharey='col')
axs[0].hist(lh_val, bins=100, alpha=0.5, label='val')
axs[0].hist(lh_attack, bins=100, alpha=0.5, label='attack')
axs[0].set_title('Likelihood')
axs[1].hist(H_val, bins=100, alpha=0.5, label='val')
axs[1].hist(H_attack, bins=100, alpha=0.5, label='attack')
axs[1].set_title('Entropy')
for ax in axs.ravel(): 
    ax.set(yscale='log')

# axs.set_title('Entropy')
# axs.legend()

plt.show()

df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1)

# Assuming that dataframes df1 and df2 are already defined
display_side_by_side(df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1), df1.style.background_gradient(subset=df1.columns, cmap=cmap, vmin=0.5, vmax=1))

np.min(entropy_val), np.max(entropy_val)

KDE_scott[(dim,n)].factor

KDE_scott[(dim,n)].evaluate(entropy_val), KDE_scott[(dim,n)].evaluate(entropy_attack)

KDE_scott[(dim,n)].pdf( np.min(entropy_val)), KDE_scott[(dim,n)].pdf(np.max(entropy_val))

np.min(KDE_scott[(dim,n)].pdf(entropy_val)), np.max(KDE_scott[(dim,n)].pdf(entropy_val))

kde_ = KDE_scott[(dim,n)]

entropy_val[0], entropy_attack[0]

likelihood = -np.log(kde_.pdf(entropy_val[0]))
likelihood

likelihood = -np.log(kde_.pdf(entropy_attack[0]))
likelihood


# Now we need clustering labels to compute predict_proba based on weights

#weights_eq = [1/3, 1/3, 1/3]
# weights_24 = [1, 0, 0, 0, 0]
# weights_26 = [0, 1, 0, 0, 0]
#weights_28 = [1, 0, 0]
weights_0 = [0, 1, 0]
weights_3 = [0, 0, 1]
# weights_28 = [1, 0, 0]
weights_0_3 = [ 0, 0.5, 0.5]

# w_list = [weights_eq, weights_3, weights_0, weights_28]

w_dict = {#'equal' : weights_eq, 
          #'feat-28' : weights_28,
          'clas-3' : weights_3, 
          'clas-0' : weights_0, 
          'clas-0-3' : weights_0_3, 
         }

weights_28 = [1, 0, 0,]
weights_0 = [0, 1, 0,]
weights_3 = [0, 0, 1,]
weights_eq = [1/3, 1/3, 1/3]
weights_0_3 = [0, 0.5, 0.5]

# w_list = [weights_eq, weights_3, weights_0, weights_28]

w_dict = {'equal' : weights_eq, 
          # 'feat-24' : weights_24,
          # 'feat-26' : weights_26,
          'feat-28' : weights_28,
          # 'feat-all' : weights_feat, 
          'clas-0-3' : weights_0_3, 
         }

num_clusters = [50, 100, 120, 150]
dims_list = [50, 60, 70]

dim = 30
n =150

weights = weights_0

ep = empirical_posterior[(dim, n)]

clv = clustering_labels_val_[(dim, n)]
cla = clustering_labels_attack[(dim, n)]

pv = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=clv, 
                   n_classes=num_classes)

pa = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=cla, 
                   n_classes=num_classes)

entropy_val = H(pv,axis=1)
entropy_attack = H(pa,axis=1)


fig, axs = plt.subplots(1, figsize=(10,8))
axs.hist(entropy_val, bins=50, density=True, alpha=0.5, label='val')
#axs.hist(entropy_attack, bins=30, density=False, alpha=0.5, label='attack')
axs.set_title('entropy')
axs.legend()

entropy_val.shape

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            # clt = clustering_labels_train[(dim, n)]
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            # pt = predict_proba(weights=weights, 
            #                    empirical_posterior=ep, 
            #                    clustering_labels=clt, 
            #                    n_classes=num_classes)

            # prob_train[(dim, n)] = pt

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
    data = []
    for dim in dims_list:
        for n in num_clusters:

            entropy_val = H(prob_val[(dim, n)],axis=1)
            entropy_attack = H(prob_attack[(dim, n)],axis=1)
            
            original_labels = np.zeros(len(entropy_val))
            attack_labels = np.ones(len(entropy_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([entropy_val, entropy_attack])
            
            auc_ = AUC(y_true=y_true, y_score=y_score)

            data.append((dim, n, auc_))

    if key=='equal': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)

dim = 10
n = 15
weights = [1, 0, 0]
ep = empirical_posterior[(dim, n)]
            
clv = clustering_labels_val_[(dim, n)]
cla = clustering_labels_attack[(dim, n)]

pv = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=clv, 
                   n_classes=num_classes)

pa = predict_proba(weights=weights, 
                   empirical_posterior=ep, 
                   clustering_labels=cla, 
                   n_classes=num_classes)

H_val = H(pv,axis=1)
H_attack = H(pa,axis=1)

H_val.mean(), H_attack.mean()

H_val.std(), H_attack.std()

H_attack.shape

fig, axs = plt.subplots(1,figsize=(10,8))
axs.hist(H_val, bins=50, density=True, alpha=0.5, label='val')
axs.hist(H_attack, bins=100, density=True, alpha=0.5, label='attack')
axs.set_title('Entropy')
axs.legend()

plt.show()

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            
            clv = clustering_labels_val_[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
    data = []
    for dim in dims_list:
        for n in num_clusters:

            max_val = np.max(prob_val[(dim, n)],axis=1)
            max_attack = np.max(prob_attack[(dim, n)],axis=1)
            
            original_labels = np.zeros(len(entropy_val))
            attack_labels = np.ones(len(entropy_attack))

            y_true = np.concatenate([original_labels, attack_labels])

            y_score = np.concatenate([max_val,max_attack])
            
            auc_ = AUC(y_true=y_true, y_score=y_score, average='macro', multi_class='ovo')

            data.append((dim, n, auc_))

    if key=='equal': # starting building results dataframe
        df = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df.set_index(['dim', 'n_clusters'], inplace=True)
        df.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df = df.join(df_)

df.style.background_gradient(subset=df.columns, cmap=cmap, vmin=0.5, vmax=1)

dims_list, num_clusters, dict_SVD.keys()

# dict_peephole_train = {} 
# dict_peephole_val = {} 
# empirical_posterior = {} 
# distances_prob_train = {}
# distances_prob_val = {}
# clustering_labels_train = {}
# clustering_labels_val = {} 

dict_peephole_attack = {}
distances_prob_attack = {} 
clustering_labels_attack = {} 

for dim in dims_list:
    
    # 1. get peepholes train 
    n_ = 10
    n_clusters_ref = {'feat-24':n_, 'feat-26':n_, 'feat-28':n_, 'clas-0': n_, 'clas-3': n_, }
    # dict_peephole_train_ref = get_dict_peephole_train(dim=dim,
    #                                                   dict_activations_train=dict_activations_train,
    #                                                   n_clusters=n_clusters_ref,
    #                                                   dict_SVD=dict_SVD)
    
    dict_peephole_train_ref = dict_peephole_train[(dim, n_)]
    
    # 2. get peepholes val
    # dict_peephole_val_ref = get_dict_peephole_val(dim=dim, 
    #                                               dict_activations_val=dict_activations_val, 
    #                                               dict_peephole_train=dict_peephole_train_ref,
    #                                               n_clusters=n_clusters_ref,
    #                                               dict_SVD=dict_SVD)
    
    dict_peephole_attack_ref = get_dict_peephole_val(dim=dim, 
                                                     dict_activations_val=dict_activations_attack, 
                                                     dict_peephole_train=dict_peephole_train_ref, # vedere
                                                     n_clusters=n_clusters_ref,
                                                     dict_SVD=dict_SVD)
    
    for n in num_clusters:
        
        n_clusters_ref = {'feat-24':n, 'feat-26':n, 'feat-28':n, 'clas-0': n, 'clas-3': n, }
        
        # 3. get updated peepholes train 
        # dict_peephole_train[(dim, n)] = get_clustering_config(dict_peephole_train=dict_peephole_train_ref,
        #                                                       n_clusters=n_clusters
        #                                                      )

        # 4. get updated peepholes val
        # dict_peephole_val[(dim, n)] = {'peephole': dict_peephole_val_ref['peephole'],
        #                               'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        dict_peephole_attack[(dim, n)] = {'peephole': dict_peephole_attack_ref['peephole'],
                                          'clustering_config': dict_peephole_train[(dim, n)]['clustering_config']}
        
        
        # 5. get empirical posteriors
        # empirical_posterior[(dim, n)] = fit_empirical_posteriors(dict_activations_train=dict_activations_train, 
        #                                                          dict_peephole_train=dict_peephole_train[(dim, n)], 
        #                                                          n_classes=num_classes)
        
        # 6. get membership probability for training set
        # distances_prob_train[(dim, n)] = get_distances_prob(dict_peephole_val=None,
        #                                                     dict_peephole_train=dict_peephole_train[(dim, n)],
        #                                                     method=method, 
        #                                                     dim=dim)
        
        # 7. get membership probability for training set
        # distances_prob_val[(dim, n)] = get_distances_prob(dict_peephole_val=dict_peephole_val[(dim, n)],
        #                                                   dict_peephole_train=dict_peephole_train[(dim, n)],
        #                                                   method=method, 
        #                                                   dim=dim)
        
        distances_prob_attack[(dim, n)] = get_distances_prob(dict_peephole_val=dict_peephole_attack[(dim, n)],
                                                             dict_peephole_train=dict_peephole_train[(dim, n)],
                                                             method=method, 
                                                             dim=dim)
        
        # 8. combine clustering labels train
        # clustering_labels_t = []
        # distances_prob = distances_prob_train[(dim, n)]
        # for element in zip(*distances_prob):
        #     ll = element
        #     clustering_labels_t.append(ll)
        # clustering_labels_train[(dim, n)] = clustering_labels_t
            
        # 9. combine clustering labels val
        # clustering_labels_v = []
        # distances_prob = distances_prob_val[(dim, n)]
        # for element in zip(*distances_prob):
        #     ll = element
        #     clustering_labels_v.append(ll)
        # clustering_labels_val[(dim, n)] = clustering_labels_v
        
        clustering_labels_a = []
        distances_prob = distances_prob_attack[(dim, n)]
        for element in zip(*distances_prob):
            ll = element
            clustering_labels_a.append(ll)
        clustering_labels_attack[(dim, n)] = clustering_labels_a
        
    # name = f'dict_peephole_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, dict_peephole_train)
    # name = f'dict_peephole_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, dict_peephole_val)
    # name = f'empirical_posterior-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, empirical_posterior)
    # name = f'distances_prob_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, distances_prob_train)
    # name = f'distances_prob_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, distances_prob_val)
    # name = f'clustering_labels_train-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, clustering_labels_train)
    # name = f'clustering_labels_val-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    # save_res(name, clustering_labels_val)
    
    name = f'_dict_peephole_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, dict_peephole_attack)
    name = f'_distances_prob_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, distances_prob_attack)
    name = f'_clustering_labels_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    save_res(name, clustering_labels_attack)

dict_peephole_attack_ = []

distances_prob_attack_ = []

clustering_labels_attack_ = []

for dim in dims_list:

    name = f'dict_peephole_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    dict_peephole_attack_.append(load_res(name))

    name = f'distances_prob_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    distances_prob_attack_.append(load_res(name))
    
    name = f'clustering_labels_attack-dim={dim}-method={method}-dataset=CIFAR100.pkl'
    clustering_labels_attack_.append(load_res(name))

dict_peephole_attack = {k: v for d in dict_peephole_attack_ for k, v in d.items()}
distances_prob_attack = {k: v for d in distances_prob_attack_ for k, v in d.items()} 
clustering_labels_attack = {k: v for d in clustering_labels_attack_ for k, v in d.items()}

dict_peephole_attack[(10,10)]['peephole'].keys()

for key, weights in w_dict.items():
    
    prob_train = {}
    prob_val = {}
    prob_attack = {}

    for dim in tqdm(dims_list):
        for n in num_clusters:

            ep = empirical_posterior[(dim, n)]
            clt = clustering_labels_train[(dim, n)]
            clv = clustering_labels_val[(dim, n)]
            cla = clustering_labels_attack[(dim, n)]

            pt = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clt, 
                               n_classes=num_classes)

            prob_train[(dim, n)] = pt

            pv = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=clv, 
                               n_classes=num_classes)

            prob_val[(dim, n)] = pv

            pa = predict_proba(weights=weights, 
                               empirical_posterior=ep, 
                               clustering_labels=cla, 
                               n_classes=num_classes)

            prob_attack[(dim, n)] = pa
            
    data = []
    data_ = []
    for dim in dims_list:
        for n in num_clusters:
            
            entropy_train = H(prob_train[(dim, n)])
            entropy_val = H(prob_val[(dim, n)])
            entropy_attack = H(prob_attack[(dim, n)])

            threshold = []
            list_val_entropy_ = []
            list_attack_entropy_ = []
            
            array = np.arange(0,100,0.1)
            
            for i in array:
            
                perc = np.percentile(entropy_train, 100-i)
                
                threshold.append(perc)
                idx_val = np.argwhere(entropy_val<perc)[:,0]
                idx_attack = np.argwhere(entropy_attack<perc)[:,0]
                percentage_val = len(idx_val)/len(entropy_val)
                percentage_attack = len(idx_attack)/len(entropy_attack)
                list_val_entropy_.append(percentage_val)
                list_attack_entropy_.append(percentage_attack)

            area_val = np.trapz(list_val_entropy_)
            area_attack = np.trapz(list_attack_entropy_)
            
            # original_labels = np.zeros(len(entropy_val))
            # attack_labels = np.ones(len(entropy_attack))

            # y_true = np.concatenate([original_labels, attack_labels])

            # y_score = np.concatenate([entropy_val, entropy_attack])
            
            # auc_ = AUC(y_true=y_true, y_score=y_score)

            data.append((dim, n, area_attack))
            data_.append((dim, n, area_val))

    if key=='equal': # starting building results dataframe
        df_area_attack = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_area_attack.set_index(['dim', 'n_clusters'], inplace=True)
        df_area_attack.sort_index(inplace=True)
    else: # concatenate with previous results
        df_ = pd.DataFrame(data, columns=['dim', 'n_clusters', key])
        df_.set_index(['dim', 'n_clusters'], inplace=True)
        df_.sort_index(inplace=True)
        df_area_attack = df_area_attack.join(df_)

    if key=='equal': # starting building results dataframe
        df_area_val = pd.DataFrame(data_, columns=['dim', 'n_clusters', key])
        df_area_val.set_index(['dim', 'n_clusters'], inplace=True)
        df_area_val.sort_index(inplace=True)
    else: # concatenate with previous results
        df0_ = pd.DataFrame(data_, columns=['dim', 'n_clusters', key])
        df0_.set_index(['dim', 'n_clusters'], inplace=True)
        df0_.sort_index(inplace=True)
        df_area_val = df_area_val.join(df0_)


df_area_attack.style.background_gradient(subset=df.columns, cmap=cmap, vmin=300, vmax=1000)



