import torchvision.datasets as datasets
import torchvision.transforms as tf
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import model.wideresnet as wideresnet
import model.wideresnet2 as wideresnet2


def load_partial_model(model,saved_state_dict,exclusion=[]):
    if isinstance(exclusion,list):
        exclusion = tuple(exclusion)
    
    if isinstance(exclusion,str) or isinstance(exclusion,tuple):
        new_params = model.state_dict().copy()
        for key in new_params:
            if not key.startswith(exclusion):
                new_params[key] = saved_state_dict[key]
        model.load_state_dict(new_params)

def create_model(num_classes,model_type):
    if model_type == 'wide22':
        model = wideresnet.WideResNet(22, num_classes, widen_factor=1, dropRate=0.0, leakyRate=0.1)
    elif model_type == 'wide28':
        model = wideresnet.WideResNet(28, num_classes, widen_factor=2, dropRate=0.0, leakyRate=0.1) 
    elif model_type == 'wide28_2':
        model = wideresnet2.WideResNet(num_classes)
    return model


def prepare_mnist(num_classes=10,root='../mnist_data'):
    # normalize data    

        
    # load train data
    train_dataset = datasets.MNIST(
                        root=root, 
                        train=True,  
                        download=True)
    
    # load test data
    test_dataset = datasets.MNIST(
                        root=root, 
                        train=False, 
                        download=True)

    train_idx = train_dataset.targets < num_classes
    
    
    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = train_dataset.targets[train_idx]

    test_idx = test_dataset.targets < num_classes

    test_dataset.data = test_dataset.data[test_idx]
    test_dataset.targets = test_dataset.targets[test_idx]
    
    return train_dataset, test_dataset


def prepare_svhn(num_classes=10,root='../svhn_data'):    
        
    # load train data
    train_dataset = datasets.SVHN(
                        root=root, 
                        split='train', 
                        transform=tf.Compose([tf.RandomCrop(32,padding=2),tf.ToTensor(),tf.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]),  
                        download=True)
    
    # load test data
    test_dataset = datasets.SVHN(
                        root=root, 
                        split='test',  
                        transform=tf.Compose([tf.ToTensor(),tf.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]),
                        download=True)

    train_idx = train_dataset.labels < num_classes

    
    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.labels = train_dataset.labels[train_idx]

    test_idx = test_dataset.labels < num_classes

    test_dataset.data = test_dataset.data[test_idx]
    test_dataset.labels = test_dataset.labels[test_idx]
    
    return train_dataset, test_dataset


def prepare_cifar10(num_classes=10,root='../cifar10_data'):
        
    # load train data
    train_dataset = datasets.CIFAR10(
                        root=root, 
                        train=True,   
                        download=True)
    
    # load test data
    test_dataset = datasets.CIFAR10(
                        root=root, 
                        train=False, 
                        download=True)

    train_idx = np.array(train_dataset.targets) < num_classes

    
    train_dataset.data = train_dataset.data[train_idx]
    train_dataset.targets = np.array(train_dataset.targets)[train_idx].tolist()

    test_idx = np.array(test_dataset.targets) < num_classes

    test_dataset.data = test_dataset.data[test_idx]
    test_dataset.targets = np.array(test_dataset.targets)[test_idx].tolist()
    
    return train_dataset, test_dataset


def prepare_cifar10_zca(num_classes=10,root='../cifar10_zca_data/cifar10_gcn_zca_v2.npz'):

    data = np.load(root)
        
    train_data = data['train_x'].astype(np.float32)
    train_labels = data['train_y'].astype(int)
    test_data = data['test_x'].astype(np.float32) 
    test_labels = data['test_y'].astype(int) 

    train_idx = train_labels < num_classes
    test_idx = test_labels < num_classes

    train_data = train_data[train_idx]
    train_labels = train_labels[train_idx]
    test_data = test_data[test_idx]
    test_labels = test_labels[test_idx]
    
    return train_data, train_labels,test_data,test_labels



def prepare_train_data(labeled_size,unlabeled_size,data_type,num_classes=10,random_state=42,shuffle=True):
    if data_type == 'mnist':
        train_dataset, test_dataset = prepare_mnist(num_classes)
    elif data_type == 'svhn':
        train_dataset, test_dataset = prepare_svhn(num_classes)        
    elif data_type == 'cifar10':
        train_dataset, test_dataset = prepare_cifar10(num_classes)        
    elif data_type == 'cifar10_zca':
        data,targets,_,_ = prepare_cifar10_zca(num_classes)
    

    if data_type != 'cifar10_zca':
        data = train_dataset.data

    if data_type == 'mnist':
        targets = train_dataset.targets
    elif data_type == 'svhn':
        targets = train_dataset.labels
    elif data_type == 'cifar10':
        targets = np.array(train_dataset.targets)

    train_indices = np.arange(len(targets))

    #labeled_indices, unlabeled_indices = train_test_split(train_indices,train_size=labeled_size,test_size=unlabeled_size,stratify=targets,random_state=random_state,shuffle=True)
    labeled_indices, others = sample_dataset(targets,labeled_size,num_classes,random_state=random_state,return_others=True)
    
    labeled_x = data[labeled_indices]
    labeled_y = targets[labeled_indices]

    data = data[others]
    targets = targets[others]

    unlabeled_indices = sample_dataset(targets,unlabeled_size,num_classes,random_state=random_state)

    np.random.seed(random_state)
    np.random.shuffle(unlabeled_indices)

    unlabeled_x = data[unlabeled_indices]
    unlabeled_y = targets[unlabeled_indices]

    return labeled_x,labeled_y,unlabeled_x,unlabeled_y


def prepare_train_data2(labeled_size,unlabeled_size,data_type,num_classes=10,random_state=42,shuffle=True):
    if data_type == 'mnist':
        train_dataset, test_dataset = prepare_mnist(num_classes)
    elif data_type == 'svhn':
        train_dataset, test_dataset = prepare_svhn(num_classes)        
    elif data_type == 'cifar10':
        train_dataset, test_dataset = prepare_cifar10(num_classes)        
    elif data_type == 'cifar10_zca':
        data,targets,_,_ = prepare_cifar10_zca(num_classes)
    

    if data_type != 'cifar10_zca':
        data = train_dataset.data

    if data_type == 'mnist':
        targets = train_dataset.targets
    elif data_type == 'svhn':
        targets = train_dataset.labels
    elif data_type == 'cifar10':
        targets = np.array(train_dataset.targets)

    train_indices = np.arange(len(targets))

    np.random.seed(random_state)
    np.random.shuffle(train_indices)

    labeled_indices = train_indices[:labeled_size]
    unlabeled_indices = train_indices[labeled_size:labeled_size+unlabeled_size]
    
    labeled_x = data[labeled_indices]
    labeled_y = targets[labeled_indices] 

    unlabeled_x = data[unlabeled_indices]
    unlabeled_y = targets[unlabeled_indices]

    return labeled_x,labeled_y,unlabeled_x,unlabeled_y


def prepare_train_data_all(labeled_size,data_type,num_classes=10,random_state=42,shuffle=True):
    if data_type == 'mnist':
        train_dataset, test_dataset = prepare_mnist(num_classes)
    elif data_type == 'svhn':
        train_dataset, test_dataset = prepare_svhn(num_classes)        
    elif data_type == 'cifar10':
        train_dataset, test_dataset = prepare_cifar10(num_classes)        
    elif data_type == 'cifar10_zca':
        data,targets,_,_ = prepare_cifar10_zca(num_classes)
    

    if data_type != 'cifar10_zca':
        data = train_dataset.data

    if data_type == 'mnist':
        targets = train_dataset.targets
    elif data_type == 'svhn':
        targets = train_dataset.labels
    elif data_type == 'cifar10':
        targets = np.array(train_dataset.targets)

    train_indices = np.arange(len(targets))

    #labeled_indices, unlabeled_indices = train_test_split(train_indices,train_size=labeled_size,test_size=unlabeled_size,stratify=targets,random_state=random_state,shuffle=True)
    labeled_indices, unlabeled_indices = sample_dataset(targets,labeled_size,num_classes,random_state=random_state,return_others=True)
    
    labeled_x = data[labeled_indices]
    labeled_y = targets[labeled_indices]

    unlabeled_x = data[unlabeled_indices]
    unlabeled_y = targets[unlabeled_indices]

    return labeled_x,labeled_y,unlabeled_x,unlabeled_y



def sample_dataset(targets, k, n_classes,random_state=42,return_others=False):
    card = k / n_classes
    indices = np.zeros(k,dtype=np.int64)
    others = np.zeros(len(targets) - k,dtype=np.int64)
    targets = torch.LongTensor(targets)

    other_count = 0
    
    for i in range(n_classes):
        size = int((i + 1) * card) -  int(i * card)
        class_items = (targets == i).nonzero().view(-1).numpy()
        size = min(size,class_items.shape[0])
        #print(class_items)
        np.random.seed(random_state)
        np.random.shuffle(class_items)
        indices[int(i * card): int((i + 1) * card)] = class_items[:size]
        if size < class_items.shape[0]:
            size2 = class_items.shape[0] - size
            others[other_count : other_count + size2] = class_items[size:]
            other_count += size2        

    if return_others:
        return indices,others
    else:
        return indices
    

    