import numpy as np
import sys
from os.path import dirname, abspath, join
from emukit.examples.profet.meta_benchmarks import meta_xgboost, meta_svm, meta_fcnet

def xgboost_function(x0, x1, x2, x3, x4, x5, x6, x7, label):
    path_to_files = join(dirname(dirname(abspath(__file__))),"data/profet_data")
    function_family = "xgboost"
    function_id = label
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    fcn, parameter_space = meta_xgboost(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    X = np.array([[x0, x1, x2, x3, x4, x5, x6, x7]])
    result = fcn(X)
    try:
        y, c = result # function value, cost for all functions except forrester
    except ValueError:
        y = result
    return float(y[0,0])

def svm_function(x0, x1, label):
    path_to_files = join(dirname(dirname(abspath(__file__))),"data/profet_data")
    function_family = "svm"
    function_id = label
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    fcn, parameter_space = meta_svm(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    X = np.array([[x0, x1]])
    result = fcn(X)
    try:
        y, c = result # function value, cost for all functions except forrester
    except ValueError:
        y = result
    return float(y[0,0])

def fcnet_function(x0, x1, x2, x3, x4, x5, label):
    path_to_files = join(dirname(dirname(abspath(__file__))),"data/profet_data")
    function_family = "fcnet"
    function_id = label
    fname_objective = "%s/samples/%s/sample_objective_%d.pkl" % (path_to_files, function_family, function_id)
    fname_cost="%s/samples/%s/sample_cost_%d.pkl" % (path_to_files, function_family, function_id)

    fcn, parameter_space = meta_fcnet(fname_objective=fname_objective, fname_cost=fname_cost, noise=False)
    X = np.array([[x0, x1, x2, x3, x4, x5]])
    result = fcn(X)
    try:
        y, c = result # function value, cost for all functions except forrester
    except ValueError:
        y = result
    return float(y[0,0])