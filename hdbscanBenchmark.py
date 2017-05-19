
# coding: utf-8

# ## A runtime comparison of HDBSCAN\* and DBSCAN
# In this notebook we will attempt an apples to apples comparison between our python implementation of HDBSCAN\* and sklearns implementation of DBSCAN.  
# 
# As mentioned detailed in \cite{paper} HDBSCAN can be thought of as computing DBSCAN for all values of epsilon then taking a variable height cut through the corresponding dendrogram.  Thus, these algorithms serve very similar functions.
# 
# Like all tree based algorithms run time and run time complexity are data dependent.  As indicated in \cite{} this raises significant difficulties when benchmarking algorithms or implementations.  Our interest is in demonstrating the comparability of scaling performance for these algorithms under the assumption both algorithms being tree based they should have similar performance changes under different data distributions.  As such, we will examine the run time behaviour of both algorithms with respect to a fairly simple data set.  
# 
# Rough transition
# 
# The difficulty with DBSCAN run time comparisons with default perameters is that very small values of $\varepsilon$ will return few or no core points which results in a very fast run with virually all the data being relagated to background noise.  To circuvent this problem and we will perform a search over the parameter space of DBSCAN in order to find the paramaters which best match our HDBSCAN\* results on a particular data set.  Then we will benchmark the training of a DBSCAN model with those specific parameters.  Though, in practice, a user may not know the optimal parameter values for DBSCAN we will exclude that from this experiment.  For this simple scaling experiment our data will consist of mixtures of a gaussian distributions laid down within a fixed diameter box.  We use varible numbers of constant variance gaussian balls for simplicity and to not unfairly penalize DBSCAN.  We vary dimension, number of clusters and number of data points to determine their effect on run time.
# 
# Our paremter search will be conducted using gaussian process optimization.  The adjusted rand index between DBSCAN clusterings and a pre-established HDBSCAN\* clustering will be used as the objective function to be maximized.  
# 
#  We make use of sklearns make_blobs function for generatig this data.
# 
# 

# In[1]:

#Data handling and exploration
import pandas as pd
import numpy as np
#Clustering and visualization
import sklearn.datasets
import sklearn.cluster
import sklearn.manifold
import sklearn.metrics
import hdbscan
from timeit import default_timer as timer


#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

def relabelOutliers(labels):
    """
    Args:
    labels (array of ints): output list of cluster id's (e.g. from hdbscan or dbscan) with outliers labeled as -1
    Returns:
    array of ints: an array corresponding to labels with each -1 relabeled to a one up number greater than
    the previous arrays max value.
    This has the effect of allocating each outlier to it's own cluster
    """
    lab = labels
    lab[lab == -1] = list(range(sum(lab == -1))) + max(lab)
    return (lab)

def clusterAdjustedRandScore(labels1, labels2):
    """
    This assumes that outlier points will be labelled with -1.
    These points should be considered to lie each within their own cluster
    and not be treated as being a single large background cluster.
    We relabel background points before computing the standard adjusted rand index.d
    """
    return (sklearn.metrics.adjusted_rand_score(relabelOutliers(labels1), relabelOutliers(labels2)))
# In[3]:

import pkg_resources
print('hdbscan version: {}'.format(pkg_resources.get_distribution("hdbscan").version))
print('pandas version: {}'.format(pkg_resources.get_distribution("pandas").version))
print('skopt version: {}'.format(pkg_resources.get_distribution("skopt").version))
print('sklearn version: {}'.format(pkg_resources.get_distribution("scikit-learn").version))


results = pd.read_csv("partialBenchmarkResults.csv")
print("In reality test {} combinations".format(results.shape[0]))


# 
# We see that not all of the results are complete.  This due to the computational expense of the dbscan paramter search over a particularly difficult parameter selection.  For very large data set sizes in two dimensions our gaussians will often overlap quite heavily.  In this case the most natural persistent cluster from the perspective of hdbscan.  The default parameters for hdbscan specifically restrict the number of clusters returned via the persistence calculation to be $\ge$ 2.  The next most persistent clusterings in these cases are a multitute of very small variably dense pockets created by random perturbations within the gaussian distributions of data.  This is a very difficult clustering to fit with traditional dbscan.  Further on such a densly packed data set a large epsilon parameter (see our paramter range) can induces a graph of almost $n^2$ size which seriously degrades the performance of dbscan for a number of paramters combinations.  In these cases both the memory and time complexity of the dbscan iterations skyrocketed.   At the time of this writing a small number of our optimizations have been running for weeks on a very large memory SMP system.  
# 
# One could vary the dbscan parameter search space for varying combinations of data set parameters but we felt that would bias the experiment.  Instead we felt it was informative to include the original experiment (though incomplete) as is.  

results.sort_values('hdbscan_time', inplace=True)


with open('hdbscanDBscanLaptopResults.csv', mode='w') as file:
    colNames = ['seed', 'size', 'dimension', 'clusters', 'rep','hdbscan_time', 'dbscan_time',
                'adjusted_rand', 'hdbscan_min_samples','hdbscan_min_cluster_size', 'dbscan_epsilon',
                'dbscan_min_samples','dbscan_num_cluster','hdbscan_num_clusters', 'cluster_rand',
                'new_hdbscan_time', 'new_dbscan_time',]
    file.write(",".join(colNames)+"\n")

    for i in range(results.shape[0]):
    #for index, instance in results.iterrows():
        #Get the paramters necessary to reproduce the data set
        instance = results.iloc[i]
        size=np.int(instance['size'])
        dataset_dimension=np.int(instance.dimension)
        dataset_n_clusters=np.int(instance.clusters)
        data, data_labels = sklearn.datasets.make_blobs(n_samples=size,
                                                   n_features=dataset_dimension,
                                                   centers=dataset_n_clusters,random_state=np.int(instance.seed))

        #Reproduce and time the hdbscan clustering
        hdbscan_min_samples = np.int(instance.hdbscan_min_samples)
        hdbscan_min_cluster_size = np.int(instance.hdbscan_min_cluster_size)
        start = timer()
        labels = hdbscan.HDBSCAN(min_samples=hdbscan_min_samples,
                                    min_cluster_size=hdbscan_min_cluster_size).fit_predict(data)
        end = timer()
        new_hdbscan_time = end - start
        hdbscan_num_clusters = len(np.unique(labels))-1

        #Time the training of a dbscan model with parameters found via attempting to match the hdbscan clustering results
        # Recall that we measured cluster similarity via the adjusted rand index and used gaussian process optimization
        # to search the dbscan parameter space.
        start = timer()
        dbscan_labels = sklearn.cluster.DBSCAN(eps=instance.dbscan_epsilon, min_samples= instance.dbscan_min_samples).fit_predict(data)
        end = timer()
        new_dbscan_time = end-start
        dbscan_num_clusters = len(np.unique(dbscan_labels))-1

        new_rand = clusterAdjustedRandScore(labels, dbscan_labels)

        print(instance.seed, size, dataset_dimension, dataset_n_clusters, instance.rep,
                   instance.hdbscan_time, instance.dbscan_time,instance.adjusted_rand,
              hdbscan_min_samples,hdbscan_min_cluster_size,instance.dbscan_epsilon,
              instance.dbscan_min_samples, dbscan_num_clusters,hdbscan_num_clusters,new_rand,
              new_hdbscan_time, new_dbscan_time,
              sep=',',file=file)
        print(i, end=',')
        if(i%10 ==0):
            print("", flush=True)


