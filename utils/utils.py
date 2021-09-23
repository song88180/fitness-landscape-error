import copy
import numpy as np
import numpy.random as nrand
from sklearn.linear_model import LinearRegression
import scipy
from scipy.linalg import hadamard
from scipy.sparse import csr_matrix
import pickle

# global variables of index data
epi_list = None
pathway_list = None
gamma_list = None
neighbor_list = None

def load_pregenerated_data(N):
    """
    Load pregenerated index files
    """
    global epi_list,pathway_list,gamma_list,neighbor_list
    if N == 5:
        with open('../index_file/epi_list_5s_all.pkl', 'rb') as f:
            epi_list = pickle.load(f)
        with open('../index_file/pathway_list_5s_all.pkl', 'rb') as f:
            pathway_list = pickle.load(f)
        with open('../index_file/gamma_list_5s_all.pkl','rb') as f:
            gamma_list = pickle.load(f)
        with open('../index_file/neighbor_list_5s_all.pkl', 'rb') as f:
            neighbor_list = np.array(pickle.load(f))
    elif N == 10:
        with open('../index_file/epi_list_10s_all.pkl', 'rb') as f:
            epi_list = pickle.load(f)
        with open('../index_file/pathway_list_10s_120000.pkl', 'rb') as f:
            pathway_list = pickle.load(f)
        with open('../index_file/gamma_list_10s_all.pkl','rb') as f:
            gamma_list = pickle.load(f)
        with open('../index_file/neighbor_list_10s_all.pkl', 'rb') as f:
            neighbor_list = np.array(pickle.load(f))
    elif N == 15:
        with open('../index_file/epi_list_15s_200000.pkl', 'rb') as f:
            epi_list = pickle.load(f)
        with open('../index_file/pathway_list_15s_240000.pkl', 'rb') as f:
            pathway_list = pickle.load(f)
        with open('../index_file/gamma_list_15s_all.pkl','rb') as f:
            gamma_list = pickle.load(f)
        with open('../index_file/neighbor_list_15s_all.pkl', 'rb') as f:
            neighbor_list = np.array(pickle.load(f))

# A primitive way of calculating N_max
#def get_N_max(landscape):
#    N = landscape.shape[1] - 1
#    N_max = 0
#    for gt in landscape:
#        seq = gt[0:N]
#        fit = gt[N]
#        flag = True
#        for i,_ in enumerate(seq):
#            seq_ = copy.deepcopy(seq)
#            seq_[i] = 1 - seq_[i]
#            tmp = ''.join(seq_.astype(int).astype(str))
#            idx = int(tmp, 2)
#            fit_ = landscape[idx,N]
#            if fit < fit_:
#                flag = False
#                break
#        if flag == True:
#            N_max += 1
#    return N_max    

# Functions to calculate different ruggedness measures

def get_N_max(landscape):
    return np.sum(np.max(landscape[neighbor_list][:,:,-1],axis=1) == landscape[neighbor_list[:,0]][:,-1])

def cal_epi(landscape):
    epi_fit_list = landscape[epi_list][:,:,-1]
    n_epi = np.sum(np.sum(epi_fit_list[:,[0,0,3,3]] > epi_fit_list[:,[1,2,1,2]],axis=1)==4)
    return n_epi/len(epi_fit_list)

def cal_r_s(landscape):
    N = landscape.shape[1] - 1
    X = landscape[:,:N]
    y = landscape[:,-1]
    reg = LinearRegression().fit(X, y) # fit_intercept default=True
    y_predict = reg.predict(landscape[:,:N])
    roughness = np.sqrt(np.mean(np.square(y - y_predict)))
    slope = np.mean(np.abs(reg.coef_))
    return roughness/slope

def cal_open_ratio(landscape):
    pathway_fit_list = landscape[pathway_list][:,:,-1]
    
    percentile20,percentile80 = np.percentile(landscape[:,-1],[20,80])
    qualified_idx = ((pathway_fit_list[:,0]<percentile20) & \
                     (pathway_fit_list[:,-1]>percentile80)) | \
                    ((pathway_fit_list[:,0]>percentile80) & \
                     (pathway_fit_list[:,-1]<percentile20))
    pathway_fit_list = pathway_fit_list[qualified_idx]
    
    total_open = np.sum(np.sum(pathway_fit_list[:,0:4]<=pathway_fit_list[:,1:5],axis=1)==pathway_fit_list.shape[1]-1)+\
    np.sum(np.sum(pathway_fit_list[:,0:4]<=pathway_fit_list[:,1:5],axis=1)==0)
    return total_open/len(pathway_fit_list)

def cal_E(landscape):
    global idx_1, phi
    N = landscape.shape[1] - 1
    W = landscape[:,-1].astype('float32')
    E = phi.dot(W)/(2**N)
    E_square = np.square(E)
    E_sum = E_square.sum()-E_square[0]
    E_1 = E_square[idx_1].sum()
    #E_2 = E_square[idx_2].sum()
    #F_2 = E_2/E_sum
    F_sum = (E_sum-E_1)/E_sum
    return F_sum

def cal_E_order(landscape):
    global idx_order, phi
    N = landscape.shape[1] - 1
    W = landscape[:,-1].astype('float32')
    E = phi.dot(W)/(2**N)
    E_square = np.square(E)
    E_sum = E_square.sum()-E_square[0]
    E_order = E_square[idx_order].sum()
    F_order = E_order/E_sum
    return F_order

def cal_gamma(landscape):
    gt_1_diff_list = landscape[gamma_list[1],-1] - landscape[gamma_list[0],-1]
    gt_2_diff_list = landscape[gamma_list[3],-1] - landscape[gamma_list[2],-1]
    cov = np.cov(gt_1_diff_list,gt_2_diff_list)[1,0]
    var = np.var(gt_1_diff_list)
    return cov/var

def cal_adptwalk_steps(landscape):
    N = landscape.shape[1] - 1
    landscape_fitness = landscape[:,-1]
    P = scipy.sparse.lil_matrix((2**N, 2**N), dtype=np.int8)
    is_absorb = np.zeros(2**N) == 1
    for i in range(2**N):
        neighbor = neighbor_list[i]
        next_idx = np.argmax(landscape_fitness[neighbor])
        P[i,neighbor[next_idx]] = 1
        if next_idx == 0:
            is_absorb[i] = True
    fittest_idx = np.argmax(landscape_fitness)
    P = P.tocsr()
    # drop the absorbing state
    Q = P[~is_absorb,:][:,~is_absorb]
    #R = P[~is_absorb,:][:,is_absorb] # calculate absorbing probability for all absorbing genotype
    #R = P[~is_absorb,fittest_idx]  # only calcualte absorbing probability for the fittest genotype
    I = scipy.sparse.identity(Q.shape[0])
    o = np.ones(Q.shape[0])
    return scipy.sparse.linalg.spsolve(I-Q, o).mean()

def cal_adptwalk_probs(landscape):
    N = landscape.shape[1] - 1
    landscape_fitness = landscape[:,-1]
    P = scipy.sparse.lil_matrix((2**N, 2**N), dtype=np.int8)
    is_absorb = np.zeros(2**N) == 1
    for i in range(2**N):
        neighbor = neighbor_list[i]
        next_idx = np.argmax(landscape_fitness[neighbor])
        P[i,neighbor[next_idx]] = 1
        if next_idx == 0:
            is_absorb[i] = True
    fittest_idx = np.argmax(landscape_fitness)
    P = P.tocsr()
    # drop the absorbing state
    Q = P[~is_absorb,:][:,~is_absorb]
    #R = P[~is_absorb,:][:,is_absorb] # calculate absorbing probability for all absorbing genotype
    R = P[~is_absorb,fittest_idx]  # only calcualte absorbing probability for the fittest genotype
    I = scipy.sparse.identity(Q.shape[0])
    return scipy.sparse.linalg.spsolve(I-Q, R).mean()

def normalize(array):
    """
    normalize values in a array
    """
    MAX = np.max(array)
    MIN = np.min(array)
    return (array - MIN)/(MAX - MIN)

def Add_Error(landscape,std):
    """
    Introduce measurement error to the FL
    """
    landscape_error = copy.deepcopy(landscape)
    landscape_error[:,-1] += np.random.normal(0,std,landscape_error.shape[0])
    landscape_error[:,-1] = normalize(landscape_error[:,-1])
    return landscape_error

phi = None
idx_1 = None

def get_ruggedness_function(metric,N,gt_code):
    """
    Return the correct ruggedness calculating function according to the input "metric"
    """
    global phi, idx_1
    if metric == 'N_max': 
        if N == 15:
            return get_N_max
        else:
            return get_N_max

    elif metric == 'r_s':
        return cal_r_s

    elif metric == 'epi':
        return cal_epi

    elif metric == 'open_ratio':
        return cal_open_ratio

    elif metric == 'E':
        phi = hadamard(2**N,dtype='float32')
        idx_1 = gt_code.sum(axis=1) == 1
        return cal_E

    elif metric == 'gamma':
        return cal_gamma

    elif metric == 'adptwalk_steps':
        return cal_adptwalk_steps

    elif metric == 'adptwalk_probs':
        return cal_adptwalk_probs
