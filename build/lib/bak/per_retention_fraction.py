measure = 'entropy'
array = np.arange(0, 1, 0.001) # setting quantiles list

for j, (layer, weight) in enumerate(list(w_dict.items())):

    for n in num_clusters:    

        for dim in dims_list: # now peep_size
            
            # fetch train and val probabilities
            prob_train = out_p_prob_train[layer][(dim, n)] # now our peepholes
            prob_val = out_p_prob_val[layer][(dim, n)] # now our peepholes
            
            threshold = []
            list_true_max_ = []
            list_false_max_ = []
            
            # compute thresholds and performance metrics
            for i in array:
                perc = np.quantile(conf_t, i)
                threshold.append(perc)
                if measure == 'max':
                    idx = np.argwhere(conf_v > perc)[:, 0]
                elif measure == 'entropy':
                    idx = np.argwhere(conf_v < perc)[:, 0]
                counter = collections.Counter(results_v[idx])
                list_true_max_.append(counter[True] / tot_true_v)
                list_false_max_.append(counter[False] / tot_false_v)
