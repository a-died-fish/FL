import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def read_gen_data(args, path):

    # Read the generated data
    images = np.load(os.path.join(path, 'images.npy'))
    labels = np.load(os.path.join(path, 'labels.npy'))

    # Shuffle the data and select the first args.fedgc_num_gen
    shuffled_indices = np.random.permutation(len(labels))
    images = images[shuffled_indices][:args.fedgc_num_gen]
    labels = labels[shuffled_indices][:args.fedgc_num_gen]
    return  images, labels


def distribute_data(args, gen_label, train_data_counts):

    client_totals = np.sum(train_data_counts, axis=1)   # number of samples per client before adding generated data

    if args.fedgc_allocation == 'equal':
        client_allocations = (np.ones(train_data_counts.shape[0]) * len(gen_label) // train_data_counts.shape[0]).astype(int)
    elif args.fedgc_allocation == 'inverse':
        client_proportions = 1.0 / client_totals
        client_proportions = client_proportions / np.sum(client_proportions) * len(gen_label)
        client_allocations = np.floor(client_proportions).astype(int)
    elif args.fedgc_allocation == 'variance': # minimize the varience between clients
        generate_number = len(gen_label)
        final_bound = -1 
        for bound in range(generate_number/len(client_totals)): # if beta is extremely low, 10% of generate_number may not be enough
            generate_used = 0
            for j in range(len(client_totals)):
                if client_totals[j]<bound:
                    generate_used += bound-client_totals
            if generate_used>generate_number:
                final_bound = bound-1
                break
        client_allocations = np.zeros(train_data_counts.shape[0])
        for i in range(train_data_counts.shape[0]):
            if client_totals[i]>=final_bound:
                client_allocations[i]=client_totals[i]
            else:
                client_allocations[i]=final_bound
        client_allocations = client_allocations.astype(int)
            

    # Split idxs according to client_allocations
    idxs = np.random.permutation(client_allocations.sum())
    gen_idxs = np.split(idxs, np.cumsum(client_allocations)[:-1])

    # Recalculate train_data_counts and client_totals
    for client_id in range(train_data_counts.shape[0]):
        for class_id in range(train_data_counts.shape[1]):
            train_data_counts[client_id][class_id] += np.sum(gen_label[gen_idxs[client_id]] == class_id)
    
    # Recalculate number of samples per client
    client_totals = np.array([len(gen_idxs[client_id])+int(client_totals[client_id]) for client_id in range(train_data_counts.shape[0])])

    print(f"{'='*20} After Adding Generation {'='*20}")  
    print(train_data_counts.astype(int))
    print(f'Number of samples per client: {client_totals}')

    return gen_idxs, client_totals, train_data_counts