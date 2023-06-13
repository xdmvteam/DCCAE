import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from linear_cca import linear_cca
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from Dcca.DCCA_train import Solver
from models import DCCA
from utils import load_data, svm_classify
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score
import gzip
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle





if __name__ == '__main__':
    device = torch.device('cuda')
    # device = torch.device('cuda:0')
    print(device)
    print("Using", torch.cuda.device_count(), "GPUs")

    # the size of the new space learned by the model (number of the new features)
    outdim_size = 10

    # size of the input for view 1 and view 2
    input_shape1 = 784
    input_shape2 = 784

    # number of layers with nodes in each one
    layer_sizes1 = [1024, 1024, 1024, outdim_size]
    layer_sizes2 = [1024, 1024, 1024, outdim_size]

    use_all_singular_values = False

    apply_linear_cca = True

    # the parameters for training the network
    learning_rate = 1e-3
    epoch_num = 20
    batch_size = 800

    reg_par = 1e-5

    checkpoint = 'DCCA_checkpoint.model'

    # the path to save the final learned features
    save_to = './new_features.gz'

    data1 = load_data('../dataset/noisymnist_view1.gz')
    data2 = load_data('../dataset/noisymnist_view2.gz')
    train1, train2 = data1[0][0], data2[0][0]
    val1, val2 = data1[1][0], data2[1][0]
    test1, test2 = data1[2][0], data2[2][0]



    model = DCCA(layer_sizes1, layer_sizes2, input_shape1,
                  input_shape2, outdim_size, use_all_singular_values, device=device).double()

    l_cca = None
    if apply_linear_cca:
        l_cca = linear_cca()
    solver = Solver(model, l_cca, outdim_size, epoch_num, batch_size,
                    learning_rate, reg_par, device=device)

    checkpoint_ = torch.load(checkpoint)
    solver.model.load_state_dict(checkpoint_)
    print("加载模型{}".format(checkpoint))

    solver.init_linear_cca(train1, train2)

    set_size = [0, train1.size(0), train1.size(0) + val1.size(0), train1.size(0) + val1.size(0) + test1.size(0)]
    loss, outputs = solver.test(torch.cat([train1, val1, test1], dim=0), torch.cat([train2, val2, test2], dim=0),
                                apply_linear_cca)
    new_data = []
    # print(outputs)
    for idx in range(3):
        new_data.append([outputs[0][set_size[idx]:set_size[idx + 1], :], outputs[1][set_size[idx]:set_size[idx + 1], :],
                         data1[idx][1]])

    # Training and testing of SVM with linear kernel on the view 1 with new features
    [test_acc, valid_acc] = svm_classify(new_data, C=0.01)
    print("Accuracy on view 1 (validation data) is:", valid_acc * 100.0)
    print("Accuracy on view 1 (test data) is:", test_acc * 100.0)
    solver.logger.info("Accuracy on view 1 (validation data) is:{:.4f}".format(valid_acc * 100.0))
    solver.logger.info("Accuracy on view 1 (test data) is:{:.4f}".format(test_acc * 100.0))


    # nmi = normalized_mutual_info_score(new_data[1][2], new_data[1][0])

    # T-SNE of Z1
    tsne = TSNE()
    Z1_test = new_data[2][0]
    y1_test = new_data[2][2]
    Z1_tsne = tsne.fit_transform(Z1_test)
    plt.scatter(Z1_tsne[:, 0], Z1_tsne[:, 1], c=y1_test, cmap='tab10')
    plt.show()

    # Saving new features in a gzip pickled file specified by save_to
    print('saving new features ...')
    f1 = gzip.open(save_to, 'wb')
    thepickle.dump(new_data, f1)
    f1.close()



















