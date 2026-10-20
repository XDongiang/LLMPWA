

import copy
import json
import logging
import logging.config
import os
import sys
import re
import time
import glob
from functools import partial
from multiprocessing import Array, Barrier, Lock, Manager, Pipe, Process, Value
from threading import Thread

import jax.experimental.host_callback as jax_exper
import jax.numpy as np
import numpy as onp
from dlib import dplex
from jax import device_put
from jax import devices as jdevices
from jax import grad, hessian, jit, jvp, value_and_grad, vmap
from jax.config import config
from jax.experimental.host_callback import id_print
from scipy.optimize import minimize


class Logger():
    def __init__(self,logger_name):
        self.logger = logging.getLogger(logger_name)
        with open("config/logconfig_fit.json","r") as config:
            LOGGING_CONFIG = json.load(config)
            logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("fit")


class args(object):
    def __init__(self):
        self.total_gpu_id = None
        self.processes_gpus = None
        self.max_processes = None
        self.max_processes_memory = None
        self.thread_gpus = None
        self.threads_in_one_gpu = None
        
        self.data_slices = None
        self.mc_slices = None
        self.mini_run = None
        
        self.bic_delete_mods = list()




class ProcessInitializers:
    def __init__(self):
        
        self.data_phif0_kk_3686 = None
        self.mc_phif0_kk_3686 = None
        self.truth_phif0_kk_3686 = None
        
        self.data_phif2_kk_3686 = None
        self.mc_phif2_kk_3686 = None
        self.truth_phif2_kk_3686 = None
        
        self.data_u_rho_r_kk_3686 = None
        self.mc_u_rho_r_kk_3686 = None
        self.truth_u_rho_r_kk_3686 = None
        
        self.data_u_rho_l_kk_3686 = None
        self.mc_u_rho_l_kk_3686 = None
        self.truth_u_rho_l_kk_3686 = None
        
        self.data_u_b_r_kk_3686 = None
        self.mc_u_b_r_kk_3686 = None
        self.truth_u_b_r_kk_3686 = None
        
        self.data_u_b_l_kk_3686 = None
        self.mc_u_b_l_kk_3686 = None
        self.truth_u_b_l_kk_3686 = None
        
        self.data_phif0_kk_3773 = None
        self.mc_phif0_kk_3773 = None
        self.truth_phif0_kk_3773 = None
        
        self.data_phif2_kk_3773 = None
        self.mc_phif2_kk_3773 = None
        self.truth_phif2_kk_3773 = None
        
        self.data_u_rho_r_kk_3773 = None
        self.mc_u_rho_r_kk_3773 = None
        self.truth_u_rho_r_kk_3773 = None
        
        self.data_u_rho_l_kk_3773 = None
        self.mc_u_rho_l_kk_3773 = None
        self.truth_u_rho_l_kk_3773 = None
        
        self.data_u_b_r_kk_3773 = None
        self.mc_u_b_r_kk_3773 = None
        self.truth_u_b_r_kk_3773 = None
        
        self.data_u_b_l_kk_3773 = None
        self.mc_u_b_l_kk_3773 = None
        self.truth_u_b_l_kk_3773 = None
        
        self.data_phi_kk = None
        self.mc_phi_kk = None
        self.truth_phi_kk = None
        
        self.data_f_kk = None
        self.mc_f_kk = None
        self.truth_f_kk = None
        
        self.data_b124_kk = None
        self.mc_b124_kk = None
        self.truth_b124_kk = None
        
        self.data_b123_kk = None
        self.mc_b123_kk = None
        self.truth_b123_kk = None
        
        
        self.wt_data_kk = None
        

        self.data_numbering = None
        self.gpu_id = None
        self.gpu_memory_limit_percentage = None

    def __repr__(self):
        return "No. " + str(self.data_numbering) + " data batch for process initializing on gpu" + str(self.gpu_id)

    def __enter__(self):
        return self

    def __exit__(self, type, value, trace):
        return self


class Process_Initializer_Generator():
    def __init__(self, data_slices=None, mc_slices=None, mini_run=None):
    
        self.data_slices = data_slices
    
        self.mc_slices = mc_slices
    
        self.mini_run = mini_run
    

    def reader_amp(self,file_name):
        amp_list = list()
        with onp.load(file_name) as amp:
            for amp_name in amp.files:
                amp_list.append((amp[amp_name])[:,0:2])
        return onp.array(amp_list)

    def data_npz(self,num):
        
        
        self.data_phif0_kk_3686 = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk_3686 = onp.array_split(self.data_phif0_kk_3686,num,axis=1)
        
        
        self.data_phif2_kk_3686 = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk_3686 = onp.array_split(self.data_phif2_kk_3686,num,axis=1)
        
        
        self.data_u_rho_r_kk_3686 = onp.load("data/real_data/u_rho_r_kk.npy")
        self.all_data_u_rho_r_kk_3686 = onp.array_split(self.data_u_rho_r_kk_3686,num,axis=1)
        
        
        self.data_u_rho_l_kk_3686 = onp.load("data/real_data/u_rho_l_kk.npy")
        self.all_data_u_rho_l_kk_3686 = onp.array_split(self.data_u_rho_l_kk_3686,num,axis=1)
        
        
        self.data_u_b_r_kk_3686 = onp.load("data/real_data/u_b_r_kk.npy")
        self.all_data_u_b_r_kk_3686 = onp.array_split(self.data_u_b_r_kk_3686,num,axis=1)
        
        
        self.data_u_b_l_kk_3686 = onp.load("data/real_data/u_b_l_kk.npy")
        self.all_data_u_b_l_kk_3686 = onp.array_split(self.data_u_b_l_kk_3686,num,axis=1)
        
        
        self.data_phif0_kk_3773 = onp.load("data/real_data/phif0_kk.npy")
        self.all_data_phif0_kk_3773 = onp.array_split(self.data_phif0_kk_3773,num,axis=1)
        
        
        self.data_phif2_kk_3773 = onp.load("data/real_data/phif2_kk.npy")
        self.all_data_phif2_kk_3773 = onp.array_split(self.data_phif2_kk_3773,num,axis=1)
        
        
        self.data_u_rho_r_kk_3773 = onp.load("data/real_data/u_rho_r_kk.npy")
        self.all_data_u_rho_r_kk_3773 = onp.array_split(self.data_u_rho_r_kk_3773,num,axis=1)
        
        
        self.data_u_rho_l_kk_3773 = onp.load("data/real_data/u_rho_l_kk.npy")
        self.all_data_u_rho_l_kk_3773 = onp.array_split(self.data_u_rho_l_kk_3773,num,axis=1)
        
        
        self.data_u_b_r_kk_3773 = onp.load("data/real_data/u_b_r_kk.npy")
        self.all_data_u_b_r_kk_3773 = onp.array_split(self.data_u_b_r_kk_3773,num,axis=1)
        
        
        self.data_u_b_l_kk_3773 = onp.load("data/real_data/u_b_l_kk.npy")
        self.all_data_u_b_l_kk_3773 = onp.array_split(self.data_u_b_l_kk_3773,num,axis=1)
        
        
        data_phi_kk = onp.load("data/real_data/phi_kk.npy")
        self.all_data_phi_kk = onp.array_split(data_phi_kk,num,axis=0)
        
        data_f_kk = onp.load("data/real_data/f_kk.npy")
        self.all_data_f_kk = onp.array_split(data_f_kk,num,axis=0)
        
        data_b124_kk = onp.load("data/real_data/b124_kk.npy")
        self.all_data_b124_kk = onp.array_split(data_b124_kk,num,axis=0)
        
        data_b123_kk = onp.load("data/real_data/b123_kk.npy")
        self.all_data_b123_kk = onp.array_split(data_b123_kk,num,axis=0)
        
        
        wt_data_kk = onp.load("data/weight/weight_kk.npy")
        self.all_wt_data_kk = onp.array_split(wt_data_kk,num,axis=0)
        

    def mc_npz(self,num):
        
        self.mc_phif0_kk_3686 = onp.load("data/mc_int/phif0_kk.npy")
        self.all_mc_phif0_kk_3686 = onp.array_split(self.mc_phif0_kk_3686,num,axis=1)
        
        self.mc_phif2_kk_3686 = onp.load("data/mc_int/phif2_kk.npy")
        self.all_mc_phif2_kk_3686 = onp.array_split(self.mc_phif2_kk_3686,num,axis=1)
        
        self.mc_u_rho_r_kk_3686 = onp.load("data/mc_int/u_rho_r_kk.npy")
        self.all_mc_u_rho_r_kk_3686 = onp.array_split(self.mc_u_rho_r_kk_3686,num,axis=1)
        
        self.mc_u_rho_l_kk_3686 = onp.load("data/mc_int/u_rho_l_kk.npy")
        self.all_mc_u_rho_l_kk_3686 = onp.array_split(self.mc_u_rho_l_kk_3686,num,axis=1)
        
        self.mc_u_b_r_kk_3686 = onp.load("data/mc_int/u_b_r_kk.npy")
        self.all_mc_u_b_r_kk_3686 = onp.array_split(self.mc_u_b_r_kk_3686,num,axis=1)
        
        self.mc_u_b_l_kk_3686 = onp.load("data/mc_int/u_b_l_kk.npy")
        self.all_mc_u_b_l_kk_3686 = onp.array_split(self.mc_u_b_l_kk_3686,num,axis=1)
        
        self.mc_phif0_kk_3773 = onp.load("data/mc_int/phif0_kk.npy")
        self.all_mc_phif0_kk_3773 = onp.array_split(self.mc_phif0_kk_3773,num,axis=1)
        
        self.mc_phif2_kk_3773 = onp.load("data/mc_int/phif2_kk.npy")
        self.all_mc_phif2_kk_3773 = onp.array_split(self.mc_phif2_kk_3773,num,axis=1)
        
        self.mc_u_rho_r_kk_3773 = onp.load("data/mc_int/u_rho_r_kk.npy")
        self.all_mc_u_rho_r_kk_3773 = onp.array_split(self.mc_u_rho_r_kk_3773,num,axis=1)
        
        self.mc_u_rho_l_kk_3773 = onp.load("data/mc_int/u_rho_l_kk.npy")
        self.all_mc_u_rho_l_kk_3773 = onp.array_split(self.mc_u_rho_l_kk_3773,num,axis=1)
        
        self.mc_u_b_r_kk_3773 = onp.load("data/mc_int/u_b_r_kk.npy")
        self.all_mc_u_b_r_kk_3773 = onp.array_split(self.mc_u_b_r_kk_3773,num,axis=1)
        
        self.mc_u_b_l_kk_3773 = onp.load("data/mc_int/u_b_l_kk.npy")
        self.all_mc_u_b_l_kk_3773 = onp.array_split(self.mc_u_b_l_kk_3773,num,axis=1)
        
        
        mc_phi_kk = onp.load("data/mc_int/phi_kk.npy")
        self.all_mc_phi_kk = onp.array_split(mc_phi_kk,num,axis=0)
        
        mc_f_kk = onp.load("data/mc_int/f_kk.npy")
        self.all_mc_f_kk = onp.array_split(mc_f_kk,num,axis=0)
        
        mc_b124_kk = onp.load("data/mc_int/b124_kk.npy")
        self.all_mc_b124_kk = onp.array_split(mc_b124_kk,num,axis=0)
        
        mc_b123_kk = onp.load("data/mc_int/b123_kk.npy")
        self.all_mc_b123_kk = onp.array_split(mc_b123_kk,num,axis=0)
        

    def truth_npz(self,num):
        
        self.truth_phif0_kk_3686 = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_truth_phif0_kk_3686 = self.truth_phif0_kk_3686[:,0:150000]
        
        self.truth_phif2_kk_3686 = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_truth_phif2_kk_3686 = self.truth_phif2_kk_3686[:,0:150000]
        
        self.truth_u_rho_r_kk_3686 = onp.load("data/mc_truth/u_rho_r_kk.npy")
        self.all_truth_u_rho_r_kk_3686 = self.truth_u_rho_r_kk_3686[:,0:150000]
        
        self.truth_u_rho_l_kk_3686 = onp.load("data/mc_truth/u_rho_l_kk.npy")
        self.all_truth_u_rho_l_kk_3686 = self.truth_u_rho_l_kk_3686[:,0:150000]
        
        self.truth_u_b_r_kk_3686 = onp.load("data/mc_truth/u_b_r_kk.npy")
        self.all_truth_u_b_r_kk_3686 = self.truth_u_b_r_kk_3686[:,0:150000]
        
        self.truth_u_b_l_kk_3686 = onp.load("data/mc_truth/u_b_l_kk.npy")
        self.all_truth_u_b_l_kk_3686 = self.truth_u_b_l_kk_3686[:,0:150000]
        
        self.truth_phif0_kk_3773 = onp.load("data/mc_truth/phif0_kk.npy")
        self.all_truth_phif0_kk_3773 = self.truth_phif0_kk_3773[:,0:150000]
        
        self.truth_phif2_kk_3773 = onp.load("data/mc_truth/phif2_kk.npy")
        self.all_truth_phif2_kk_3773 = self.truth_phif2_kk_3773[:,0:150000]
        
        self.truth_u_rho_r_kk_3773 = onp.load("data/mc_truth/u_rho_r_kk.npy")
        self.all_truth_u_rho_r_kk_3773 = self.truth_u_rho_r_kk_3773[:,0:150000]
        
        self.truth_u_rho_l_kk_3773 = onp.load("data/mc_truth/u_rho_l_kk.npy")
        self.all_truth_u_rho_l_kk_3773 = self.truth_u_rho_l_kk_3773[:,0:150000]
        
        self.truth_u_b_r_kk_3773 = onp.load("data/mc_truth/u_b_r_kk.npy")
        self.all_truth_u_b_r_kk_3773 = self.truth_u_b_r_kk_3773[:,0:150000]
        
        self.truth_u_b_l_kk_3773 = onp.load("data/mc_truth/u_b_l_kk.npy")
        self.all_truth_u_b_l_kk_3773 = self.truth_u_b_l_kk_3773[:,0:150000]
        
        
        truth_phi_kk = onp.load("data/mc_truth/phi_kk.npy")
        self.all_truth_phi_kk = truth_phi_kk[0:150000]
        
        truth_f_kk = onp.load("data/mc_truth/f_kk.npy")
        self.all_truth_f_kk = truth_f_kk[0:150000]
        
        truth_b124_kk = onp.load("data/mc_truth/b124_kk.npy")
        self.all_truth_b124_kk = truth_b124_kk[0:150000]
        
        truth_b123_kk = onp.load("data/mc_truth/b123_kk.npy")
        self.all_truth_b123_kk = truth_b123_kk[0:150000]
        

    def regular(self):
        
        self.re_phif0_kk_3686 = onp.load("data/mc_int/phif0_kk.npy")
        
        self.re_phif2_kk_3686 = onp.load("data/mc_int/phif2_kk.npy")
        
        self.re_u_rho_r_kk_3686 = onp.load("data/mc_int/u_rho_r_kk.npy")
        
        self.re_u_rho_l_kk_3686 = onp.load("data/mc_int/u_rho_l_kk.npy")
        
        self.re_u_b_r_kk_3686 = onp.load("data/mc_int/u_b_r_kk.npy")
        
        self.re_u_b_l_kk_3686 = onp.load("data/mc_int/u_b_l_kk.npy")
        
        self.re_phif0_kk_3773 = onp.load("data/mc_int/phif0_kk.npy")
        
        self.re_phif2_kk_3773 = onp.load("data/mc_int/phif2_kk.npy")
        
        self.re_u_rho_r_kk_3773 = onp.load("data/mc_int/u_rho_r_kk.npy")
        
        self.re_u_rho_l_kk_3773 = onp.load("data/mc_int/u_rho_l_kk.npy")
        
        self.re_u_b_r_kk_3773 = onp.load("data/mc_int/u_b_r_kk.npy")
        
        self.re_u_b_l_kk_3773 = onp.load("data/mc_int/u_b_l_kk.npy")
        
        
        regular_phif0_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk_3686)**2,axis=2)),axis=1)
        self.all_data_phif0_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif0_kk_3686),regular_phif0_kk_3686)
        self.all_mc_phif0_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif0_kk_3686),regular_phif0_kk_3686)
        self.all_truth_phif0_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif0_kk_3686),regular_phif0_kk_3686)
        
        regular_phif2_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk_3686)**2,axis=2)),axis=1)
        self.all_data_phif2_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif2_kk_3686),regular_phif2_kk_3686)
        self.all_mc_phif2_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif2_kk_3686),regular_phif2_kk_3686)
        self.all_truth_phif2_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif2_kk_3686),regular_phif2_kk_3686)
        
        regular_u_rho_r_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_rho_r_kk_3686)**2,axis=2)),axis=1)
        self.all_data_u_rho_r_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_rho_r_kk_3686),regular_u_rho_r_kk_3686)
        self.all_mc_u_rho_r_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_rho_r_kk_3686),regular_u_rho_r_kk_3686)
        self.all_truth_u_rho_r_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_rho_r_kk_3686),regular_u_rho_r_kk_3686)
        
        regular_u_rho_l_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_rho_l_kk_3686)**2,axis=2)),axis=1)
        self.all_data_u_rho_l_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_rho_l_kk_3686),regular_u_rho_l_kk_3686)
        self.all_mc_u_rho_l_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_rho_l_kk_3686),regular_u_rho_l_kk_3686)
        self.all_truth_u_rho_l_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_rho_l_kk_3686),regular_u_rho_l_kk_3686)
        
        regular_u_b_r_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_b_r_kk_3686)**2,axis=2)),axis=1)
        self.all_data_u_b_r_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_b_r_kk_3686),regular_u_b_r_kk_3686)
        self.all_mc_u_b_r_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_b_r_kk_3686),regular_u_b_r_kk_3686)
        self.all_truth_u_b_r_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_b_r_kk_3686),regular_u_b_r_kk_3686)
        
        regular_u_b_l_kk_3686 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_b_l_kk_3686)**2,axis=2)),axis=1)
        self.all_data_u_b_l_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_b_l_kk_3686),regular_u_b_l_kk_3686)
        self.all_mc_u_b_l_kk_3686 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_b_l_kk_3686),regular_u_b_l_kk_3686)
        self.all_truth_u_b_l_kk_3686 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_b_l_kk_3686),regular_u_b_l_kk_3686)
        
        regular_phif0_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif0_kk_3773)**2,axis=2)),axis=1)
        self.all_data_phif0_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif0_kk_3773),regular_phif0_kk_3773)
        self.all_mc_phif0_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif0_kk_3773),regular_phif0_kk_3773)
        self.all_truth_phif0_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif0_kk_3773),regular_phif0_kk_3773)
        
        regular_phif2_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_phif2_kk_3773)**2,axis=2)),axis=1)
        self.all_data_phif2_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_phif2_kk_3773),regular_phif2_kk_3773)
        self.all_mc_phif2_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_phif2_kk_3773),regular_phif2_kk_3773)
        self.all_truth_phif2_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_phif2_kk_3773),regular_phif2_kk_3773)
        
        regular_u_rho_r_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_rho_r_kk_3773)**2,axis=2)),axis=1)
        self.all_data_u_rho_r_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_rho_r_kk_3773),regular_u_rho_r_kk_3773)
        self.all_mc_u_rho_r_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_rho_r_kk_3773),regular_u_rho_r_kk_3773)
        self.all_truth_u_rho_r_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_rho_r_kk_3773),regular_u_rho_r_kk_3773)
        
        regular_u_rho_l_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_rho_l_kk_3773)**2,axis=2)),axis=1)
        self.all_data_u_rho_l_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_rho_l_kk_3773),regular_u_rho_l_kk_3773)
        self.all_mc_u_rho_l_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_rho_l_kk_3773),regular_u_rho_l_kk_3773)
        self.all_truth_u_rho_l_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_rho_l_kk_3773),regular_u_rho_l_kk_3773)
        
        regular_u_b_r_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_b_r_kk_3773)**2,axis=2)),axis=1)
        self.all_data_u_b_r_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_b_r_kk_3773),regular_u_b_r_kk_3773)
        self.all_mc_u_b_r_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_b_r_kk_3773),regular_u_b_r_kk_3773)
        self.all_truth_u_b_r_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_b_r_kk_3773),regular_u_b_r_kk_3773)
        
        regular_u_b_l_kk_3773 = 1./onp.average(onp.sqrt(onp.sum(onp.asarray(self.re_u_b_l_kk_3773)**2,axis=2)),axis=1)
        self.all_data_u_b_l_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_data_u_b_l_kk_3773),regular_u_b_l_kk_3773)
        self.all_mc_u_b_l_kk_3773 = onp.einsum("ijkl,j->ijkl",onp.array(self.all_mc_u_b_l_kk_3773),regular_u_b_l_kk_3773)
        self.all_truth_u_b_l_kk_3773 = onp.einsum("jkl,j->jkl",onp.array(self.all_truth_u_b_l_kk_3773),regular_u_b_l_kk_3773)
        
    
    def process_initializer_generator(self):
        self.data_npz(self.data_slices)
        self.mc_npz(self.mc_slices)
        self.truth_npz(self.mc_slices)
        self.regular()
        logger.info("============ w i t h  n e x t ===============")

        data_numbering = 0
        event_num = self.mini_run
        while data_numbering<event_num:
            _ = ProcessInitializers()
            # 这里 %self.data_slices 是为了在多进程的时候使程序只靠 mini_run 控制循环次数
            
            _.data_phi_kk = self.all_data_phi_kk[data_numbering%self.data_slices]
            _.mc_phi_kk = self.all_mc_phi_kk[data_numbering%self.mc_slices]
            _.truth_phi_kk = self.all_truth_phi_kk
            
            _.data_f_kk = self.all_data_f_kk[data_numbering%self.data_slices]
            _.mc_f_kk = self.all_mc_f_kk[data_numbering%self.mc_slices]
            _.truth_f_kk = self.all_truth_f_kk
            
            _.data_b124_kk = self.all_data_b124_kk[data_numbering%self.data_slices]
            _.mc_b124_kk = self.all_mc_b124_kk[data_numbering%self.mc_slices]
            _.truth_b124_kk = self.all_truth_b124_kk
            
            _.data_b123_kk = self.all_data_b123_kk[data_numbering%self.data_slices]
            _.mc_b123_kk = self.all_mc_b123_kk[data_numbering%self.mc_slices]
            _.truth_b123_kk = self.all_truth_b123_kk
            
            
            _.data_phif0_kk_3686 = self.all_data_phif0_kk_3686[data_numbering%self.data_slices]
            _.mc_phif0_kk_3686 = self.all_mc_phif0_kk_3686[data_numbering%self.mc_slices]
            _.truth_phif0_kk_3686 = self.all_truth_phif0_kk_3686
            
            _.data_phif2_kk_3686 = self.all_data_phif2_kk_3686[data_numbering%self.data_slices]
            _.mc_phif2_kk_3686 = self.all_mc_phif2_kk_3686[data_numbering%self.mc_slices]
            _.truth_phif2_kk_3686 = self.all_truth_phif2_kk_3686
            
            _.data_u_rho_r_kk_3686 = self.all_data_u_rho_r_kk_3686[data_numbering%self.data_slices]
            _.mc_u_rho_r_kk_3686 = self.all_mc_u_rho_r_kk_3686[data_numbering%self.mc_slices]
            _.truth_u_rho_r_kk_3686 = self.all_truth_u_rho_r_kk_3686
            
            _.data_u_rho_l_kk_3686 = self.all_data_u_rho_l_kk_3686[data_numbering%self.data_slices]
            _.mc_u_rho_l_kk_3686 = self.all_mc_u_rho_l_kk_3686[data_numbering%self.mc_slices]
            _.truth_u_rho_l_kk_3686 = self.all_truth_u_rho_l_kk_3686
            
            _.data_u_b_r_kk_3686 = self.all_data_u_b_r_kk_3686[data_numbering%self.data_slices]
            _.mc_u_b_r_kk_3686 = self.all_mc_u_b_r_kk_3686[data_numbering%self.mc_slices]
            _.truth_u_b_r_kk_3686 = self.all_truth_u_b_r_kk_3686
            
            _.data_u_b_l_kk_3686 = self.all_data_u_b_l_kk_3686[data_numbering%self.data_slices]
            _.mc_u_b_l_kk_3686 = self.all_mc_u_b_l_kk_3686[data_numbering%self.mc_slices]
            _.truth_u_b_l_kk_3686 = self.all_truth_u_b_l_kk_3686
            
            _.data_phif0_kk_3773 = self.all_data_phif0_kk_3773[data_numbering%self.data_slices]
            _.mc_phif0_kk_3773 = self.all_mc_phif0_kk_3773[data_numbering%self.mc_slices]
            _.truth_phif0_kk_3773 = self.all_truth_phif0_kk_3773
            
            _.data_phif2_kk_3773 = self.all_data_phif2_kk_3773[data_numbering%self.data_slices]
            _.mc_phif2_kk_3773 = self.all_mc_phif2_kk_3773[data_numbering%self.mc_slices]
            _.truth_phif2_kk_3773 = self.all_truth_phif2_kk_3773
            
            _.data_u_rho_r_kk_3773 = self.all_data_u_rho_r_kk_3773[data_numbering%self.data_slices]
            _.mc_u_rho_r_kk_3773 = self.all_mc_u_rho_r_kk_3773[data_numbering%self.mc_slices]
            _.truth_u_rho_r_kk_3773 = self.all_truth_u_rho_r_kk_3773
            
            _.data_u_rho_l_kk_3773 = self.all_data_u_rho_l_kk_3773[data_numbering%self.data_slices]
            _.mc_u_rho_l_kk_3773 = self.all_mc_u_rho_l_kk_3773[data_numbering%self.mc_slices]
            _.truth_u_rho_l_kk_3773 = self.all_truth_u_rho_l_kk_3773
            
            _.data_u_b_r_kk_3773 = self.all_data_u_b_r_kk_3773[data_numbering%self.data_slices]
            _.mc_u_b_r_kk_3773 = self.all_mc_u_b_r_kk_3773[data_numbering%self.mc_slices]
            _.truth_u_b_r_kk_3773 = self.all_truth_u_b_r_kk_3773
            
            _.data_u_b_l_kk_3773 = self.all_data_u_b_l_kk_3773[data_numbering%self.data_slices]
            _.mc_u_b_l_kk_3773 = self.all_mc_u_b_l_kk_3773[data_numbering%self.mc_slices]
            _.truth_u_b_l_kk_3773 = self.all_truth_u_b_l_kk_3773
            
            
            _.wt_data_kk = self.all_wt_data_kk[data_numbering%self.data_slices]
            
            _.data_numbering = data_numbering
            yield _
            data_numbering += 1
        return

        




class ProcessReturns:
    def __init__(self):
        self.manager = Manager()
        self._dict = self.manager.dict(
            data_numbering=None,
            process_id=None,
            parameter_values=None,
            parameter_errors=None,
            min_fcn=None,
            timer=None, 
            gpu_id=None)
    def set(self, key, value):
        self._dict[key] = value

    def get(self, key):
        return self._dict[key]

    def info(self):
        logger.info("No. {} data on GPU{} for {} s".format(
            self._dict["data_numbering"],
            self._dict["gpu_id"],
            self._dict["timer"]))
        for value, error in zip(self._dict["parameter_values"], self._dict["parameter_errors"]):
            logger.debug("value={}, error={}".format(value, error))






class PWAFunc():
    def __init__(self, cdl=None, device_id=None):
        self.device = device_id
        
        self.data_phif0_kk_3686 = device_put(np.array(cdl.data_phif0_kk_3686),device=self.device)
        self.mc_phif0_kk_3686 = device_put(np.array(cdl.mc_phif0_kk_3686),device=self.device)
        self.truth_phif0_kk_3686 = device_put(np.array(cdl.truth_phif0_kk_3686),device=self.device)
        
        self.data_phif2_kk_3686 = device_put(np.array(cdl.data_phif2_kk_3686),device=self.device)
        self.mc_phif2_kk_3686 = device_put(np.array(cdl.mc_phif2_kk_3686),device=self.device)
        self.truth_phif2_kk_3686 = device_put(np.array(cdl.truth_phif2_kk_3686),device=self.device)
        
        self.data_u_rho_r_kk_3686 = device_put(np.array(cdl.data_u_rho_r_kk_3686),device=self.device)
        self.mc_u_rho_r_kk_3686 = device_put(np.array(cdl.mc_u_rho_r_kk_3686),device=self.device)
        self.truth_u_rho_r_kk_3686 = device_put(np.array(cdl.truth_u_rho_r_kk_3686),device=self.device)
        
        self.data_u_rho_l_kk_3686 = device_put(np.array(cdl.data_u_rho_l_kk_3686),device=self.device)
        self.mc_u_rho_l_kk_3686 = device_put(np.array(cdl.mc_u_rho_l_kk_3686),device=self.device)
        self.truth_u_rho_l_kk_3686 = device_put(np.array(cdl.truth_u_rho_l_kk_3686),device=self.device)
        
        self.data_u_b_r_kk_3686 = device_put(np.array(cdl.data_u_b_r_kk_3686),device=self.device)
        self.mc_u_b_r_kk_3686 = device_put(np.array(cdl.mc_u_b_r_kk_3686),device=self.device)
        self.truth_u_b_r_kk_3686 = device_put(np.array(cdl.truth_u_b_r_kk_3686),device=self.device)
        
        self.data_u_b_l_kk_3686 = device_put(np.array(cdl.data_u_b_l_kk_3686),device=self.device)
        self.mc_u_b_l_kk_3686 = device_put(np.array(cdl.mc_u_b_l_kk_3686),device=self.device)
        self.truth_u_b_l_kk_3686 = device_put(np.array(cdl.truth_u_b_l_kk_3686),device=self.device)
        
        self.data_phif0_kk_3773 = device_put(np.array(cdl.data_phif0_kk_3773),device=self.device)
        self.mc_phif0_kk_3773 = device_put(np.array(cdl.mc_phif0_kk_3773),device=self.device)
        self.truth_phif0_kk_3773 = device_put(np.array(cdl.truth_phif0_kk_3773),device=self.device)
        
        self.data_phif2_kk_3773 = device_put(np.array(cdl.data_phif2_kk_3773),device=self.device)
        self.mc_phif2_kk_3773 = device_put(np.array(cdl.mc_phif2_kk_3773),device=self.device)
        self.truth_phif2_kk_3773 = device_put(np.array(cdl.truth_phif2_kk_3773),device=self.device)
        
        self.data_u_rho_r_kk_3773 = device_put(np.array(cdl.data_u_rho_r_kk_3773),device=self.device)
        self.mc_u_rho_r_kk_3773 = device_put(np.array(cdl.mc_u_rho_r_kk_3773),device=self.device)
        self.truth_u_rho_r_kk_3773 = device_put(np.array(cdl.truth_u_rho_r_kk_3773),device=self.device)
        
        self.data_u_rho_l_kk_3773 = device_put(np.array(cdl.data_u_rho_l_kk_3773),device=self.device)
        self.mc_u_rho_l_kk_3773 = device_put(np.array(cdl.mc_u_rho_l_kk_3773),device=self.device)
        self.truth_u_rho_l_kk_3773 = device_put(np.array(cdl.truth_u_rho_l_kk_3773),device=self.device)
        
        self.data_u_b_r_kk_3773 = device_put(np.array(cdl.data_u_b_r_kk_3773),device=self.device)
        self.mc_u_b_r_kk_3773 = device_put(np.array(cdl.mc_u_b_r_kk_3773),device=self.device)
        self.truth_u_b_r_kk_3773 = device_put(np.array(cdl.truth_u_b_r_kk_3773),device=self.device)
        
        self.data_u_b_l_kk_3773 = device_put(np.array(cdl.data_u_b_l_kk_3773),device=self.device)
        self.mc_u_b_l_kk_3773 = device_put(np.array(cdl.mc_u_b_l_kk_3773),device=self.device)
        self.truth_u_b_l_kk_3773 = device_put(np.array(cdl.truth_u_b_l_kk_3773),device=self.device)
        
        self.data_phi_kk = device_put(np.array(cdl.data_phi_kk),device=self.device)
        self.mc_phi_kk = device_put(np.array(cdl.mc_phi_kk),device=self.device)
        self.truth_phi_kk = device_put(np.array(cdl.truth_phi_kk),device=self.device)
        
        self.data_f_kk = device_put(np.array(cdl.data_f_kk),device=self.device)
        self.mc_f_kk = device_put(np.array(cdl.mc_f_kk),device=self.device)
        self.truth_f_kk = device_put(np.array(cdl.truth_f_kk),device=self.device)
        
        self.data_b124_kk = device_put(np.array(cdl.data_b124_kk),device=self.device)
        self.mc_b124_kk = device_put(np.array(cdl.mc_b124_kk),device=self.device)
        self.truth_b124_kk = device_put(np.array(cdl.truth_b124_kk),device=self.device)
        
        self.data_b123_kk = device_put(np.array(cdl.data_b123_kk),device=self.device)
        self.mc_b123_kk = device_put(np.array(cdl.mc_b123_kk),device=self.device)
        self.truth_b123_kk = device_put(np.array(cdl.truth_b123_kk),device=self.device)
        
        
        self.wt_data_kk = device_put(np.array(cdl.wt_data_kk),device=self.device)
        
    
    
    
    
    def data_likelihood_kk(self, args):
        
        kk_3686_amp_inta = np.array([-1.0])
        
        kk_3686_tha_inta = np.array([0.0])
        
        phi_mass = np.array([1.02])
        
        phi_width = np.array([0.004])
        
        kk_f980_mass = np.array([0.9672168652830319])
        
        kk_f980_g_kk = np.array([0.1253740669877132])
        
        kk_f980_rg = np.array([2.7005052659229323])
        
        kk_f980_const_3686 = np.array([0.1,args[0]]).reshape(-1,2)
        
        kk_f980_theta_3686 = np.array([0.1,args[1]]).reshape(-1,2)
        
        kk_f1270_mass = np.array([1.278468123947484])
        
        kk_f1270_width = np.array([0.18253178603942197])
        
        kk_f1270_const_3686 = np.array([args[2],args[3],args[4],args[5],args[6]]).reshape(-1,5)
        
        kk_f1270_theta_3686 = np.array([args[7],args[8],args[9],args[10],args[11]]).reshape(-1,5)
        
        kk_f2_mass = np.array([1.5150133016307599])
        
        kk_f2_width = np.array([0.09287088061436292])
        
        kk_f2_const_3686 = np.array([args[12],args[13],args[14],args[15],args[16]]).reshape(-1,5)
        
        kk_f2_theta_3686 = np.array([args[17],args[18],args[19],args[20],args[21]]).reshape(-1,5)
        
        kk_f0_mass = np.array([2.4753883485342745,2.3212816589917145,1.6998114389127226])
        
        kk_f0_width = np.array([0.09005646310357743,0.08237684207631378,0.15477458920519632])
        
        kk_f0_const_3686 = np.array([args[22],args[23],args[24],args[25],args[26],args[27]]).reshape(-1,2)
        
        kk_f0_theta_3686 = np.array([args[28],args[29],args[30],args[31],args[32],args[33]]).reshape(-1,2)
        
        kk_f0_long_mass = np.array([1.0])
        
        kk_f0_long_width = np.array([100000.0])
        
        kk_f0_long_const_3686 = np.array([args[34],args[35]]).reshape(-1,2)
        
        kk_f0_long_theta_3686 = np.array([args[36],args[37]]).reshape(-1,2)
        
        kk_f0l_mass = np.array([2.064921722894072])
        
        kk_f0l_width = np.array([0.30238434570737827])
        
        kk_f0l_const_3686 = np.array([args[38],args[39]]).reshape(-1,2)
        
        kk_f0l_theta_3686 = np.array([args[40],args[41]]).reshape(-1,2)
        
        kk_f2l_mass = np.array([1.9435205040750276,2.1783628905892916,2.3736695101913585])
        
        kk_f2l_width = np.array([0.46553324520367734,0.171365802944736,0.35232188312114704])
        
        kk_f2l_const_3686 = np.array([args[42],args[43],args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56]]).reshape(-1,5)
        
        kk_f2l_theta_3686 = np.array([args[57],args[58],args[59],args[60],args[61],args[62],args[63],args[64],args[65],args[66],args[67],args[68],args[69],args[70],args[71]]).reshape(-1,5)
        
        kk_u_rho_mass = np.array([2.397312161851163])
        
        kk_u_rho_width = np.array([0.08930983775105905])
        
        kk_u_rho_const_3686 = np.array([args[72]]).reshape(-1,1)
        
        kk_u_rho_theta_3686 = np.array([args[73]]).reshape(-1,1)
        
        kk_u_b_mass = np.array([1.8383688963975429,2.21695086990315])
        
        kk_u_b_width = np.array([0.2886776944601449,0.30820571811928366])
        
        kk_u_b_const_3686 = np.array([args[74],args[75],args[76],args[77],args[78],args[79],args[80],args[81]]).reshape(-1,4)
        
        kk_u_b_theta_3686 = np.array([args[82],args[83],args[84],args[85],args[86],args[87],args[88],args[89]]).reshape(-1,4)
        
        kk_3773_amp_inta = np.array([args[90]])
        
        kk_3773_tha_inta = np.array([args[91]])
        
        kk_f980_const_3773 = np.array([0.1,0.0812758802808357]).reshape(-1,2)
        
        kk_f980_theta_3773 = np.array([0.1,0.04219055453438748]).reshape(-1,2)
        
        kk_f1270_const_3773 = np.array([-0.001908795026008432,-0.002198862452744889,9.263590492855337e-05,0.0038573660399420315,0.004216085267475821]).reshape(-1,5)
        
        kk_f1270_theta_3773 = np.array([0.01328748133437328,0.019267847356889318,0.00823480715160893,-0.0017014797381603949,0.013688984939122168]).reshape(-1,5)
        
        kk_f2_const_3773 = np.array([0.00836160494789744,0.0022731471899687674,-0.0046768086923496405,-0.030976560693756445,0.011016487927278483]).reshape(-1,5)
        
        kk_f2_theta_3773 = np.array([0.0007651896888136746,-0.0021162876037446768,-0.005601756967141078,-0.0013397178875563952,-0.00916615679988274]).reshape(-1,5)
        
        kk_f0_const_3773 = np.array([0.006776209842051121,-0.005211775708005117,0.013982507928474695,0.007221463747811338,0.016393747705371626,0.0026018914837070166]).reshape(-1,2)
        
        kk_f0_theta_3773 = np.array([-0.019419455060917547,-0.02162941877883244,-0.016406491062264772,-0.011508544965435973,-0.02996999596857378,-0.016574134291907173]).reshape(-1,2)
        
        kk_f0_long_const_3773 = np.array([0.17112767897801573,0.05688549960596512]).reshape(-1,2)
        
        kk_f0_long_theta_3773 = np.array([-0.11230387384356917,-0.06356583331362733]).reshape(-1,2)
        
        kk_f0l_const_3773 = np.array([-0.01907412396118519,-0.014324317992303416]).reshape(-1,2)
        
        kk_f0l_theta_3773 = np.array([-0.026948437967964086,-0.009458867738581251]).reshape(-1,2)
        
        kk_f2l_const_3773 = np.array([-0.009557649920876458,-0.0031727092103564524,0.014100451734034392,0.010479172150098524,0.0003692045644304492,0.0018597973868881273,0.00457547644999926,0.003597272060697062,0.0062908372146082804,0.0035507238700523296,-0.017485005393528725,0.015226995460400562,0.0013420411659886469,0.03677091206682049,-0.0004435335025267973]).reshape(-1,5)
        
        kk_f2l_theta_3773 = np.array([-0.017692773889318524,-0.014879688649882716,0.00157904926827019,0.00635591994629108,0.006381438521152356,-0.01207283493112012,-0.0029108265764609606,0.0016613195856758618,0.011585574654273742,0.005166091307635481,-0.0645311336569186,0.003900999450326149,-0.006707506973748528,0.044428897338296275,0.005620130634310312]).reshape(-1,5)
        
        kk_u_rho_const_3773 = np.array([-0.0017169985093901598]).reshape(-1,1)
        
        kk_u_rho_theta_3773 = np.array([-0.003072954777635661]).reshape(-1,1)
        
        kk_u_b_const_3773 = np.array([0.003136380638161863,0.0020679454615716384,-0.0024864348773066037,0.005679669182781839,0.007687456697620569,-0.0014652125102298027,0.004360881343635086,0.004918464768283009]).reshape(-1,4)
        
        kk_u_b_theta_3773 = np.array([-0.00832920259713428,-0.001168027729720911,-0.011101976414934987,-0.003775730069307858,0.009518783705592684,0.011551460555613131,-0.0002691609960762099,0.0002696843264585296]).reshape(-1,4)
        
        
        data_phif0_kk_3686_BW_flatte980 = self.calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3686)
        
        data_phif2_kk_3686_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3686)
        
        data_phif2_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3686)
        
        data_phif0_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3686)
        
        data_phif0_kk_3686_BW_BW_long = self.calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3686)
        
        data_phif0_kk_3686_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3686)
        
        data_phif2_kk_3686_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3686)
        
        data_u_rho_r_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.data_phi_kk,self.data_b124_kk,self.data_u_rho_r_kk_3686)
        
        data_u_rho_l_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.data_phi_kk,self.data_b123_kk,self.data_u_rho_l_kk_3686)
        
        data_u_b_r_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.data_phi_kk,self.data_b124_kk,self.data_u_b_r_kk_3686)
        
        data_u_b_l_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.data_phi_kk,self.data_b123_kk,self.data_u_b_l_kk_3686)
        
        data_phif0_kk_3773_BW_flatte980 = self.calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3773)
        
        data_phif2_kk_3773_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3773)
        
        data_phif2_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3773)
        
        data_phif0_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3773)
        
        data_phif0_kk_3773_BW_BW_long = self.calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3773)
        
        data_phif0_kk_3773_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif0_kk_3773)
        
        data_phif2_kk_3773_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,self.data_phi_kk,self.data_f_kk,self.data_phif2_kk_3773)
        
        data_u_rho_r_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.data_phi_kk,self.data_b124_kk,self.data_u_rho_r_kk_3773)
        
        data_u_rho_l_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.data_phi_kk,self.data_b123_kk,self.data_u_rho_l_kk_3773)
        
        data_u_b_r_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.data_phi_kk,self.data_b124_kk,self.data_u_b_r_kk_3773)
        
        data_u_b_l_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.data_phi_kk,self.data_b123_kk,self.data_u_b_l_kk_3773)
        
        
        lasso_data_phif0_kk_3686_BW_flatte980 = self.lasso_calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3686)
        
        lasso_data_phif2_kk_3686_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3686)
        
        lasso_data_phif2_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3686)
        
        lasso_data_phif0_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3686)
        
        lasso_data_phif0_kk_3686_BW_BW_long = self.lasso_calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3686)
        
        lasso_data_phif0_kk_3686_BW_BW_f0_kk = self.lasso_calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3686)
        
        lasso_data_phif2_kk_3686_BW_BW_f2_kk = self.lasso_calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3686)
        
        lasso_data_u_rho_r_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.truth_phi_kk,self.truth_b124_kk,self.truth_u_rho_r_kk_3686)
        
        lasso_data_u_rho_l_kk_3686_BW_BW = self.lasso_calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.truth_phi_kk,self.truth_b123_kk,self.truth_u_rho_l_kk_3686)
        
        lasso_data_u_b_r_kk_3686_BW_BWb = self.lasso_calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.truth_phi_kk,self.truth_b124_kk,self.truth_u_b_r_kk_3686)
        
        lasso_data_u_b_l_kk_3686_BW_BWb = self.lasso_calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.truth_phi_kk,self.truth_b123_kk,self.truth_u_b_l_kk_3686)
        
        lasso_data_phif0_kk_3773_BW_flatte980 = self.lasso_calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3773)
        
        lasso_data_phif2_kk_3773_BW_flatte1270 = self.lasso_calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3773)
        
        lasso_data_phif2_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3773)
        
        lasso_data_phif0_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3773)
        
        lasso_data_phif0_kk_3773_BW_BW_long = self.lasso_calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3773)
        
        lasso_data_phif0_kk_3773_BW_BW_f0_kk = self.lasso_calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif0_kk_3773)
        
        lasso_data_phif2_kk_3773_BW_BW_f2_kk = self.lasso_calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,self.truth_phi_kk,self.truth_f_kk,self.truth_phif2_kk_3773)
        
        lasso_data_u_rho_r_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.truth_phi_kk,self.truth_b124_kk,self.truth_u_rho_r_kk_3773)
        
        lasso_data_u_rho_l_kk_3773_BW_BW = self.lasso_calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.truth_phi_kk,self.truth_b123_kk,self.truth_u_rho_l_kk_3773)
        
        lasso_data_u_b_r_kk_3773_BW_BWb = self.lasso_calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.truth_phi_kk,self.truth_b124_kk,self.truth_u_b_r_kk_3773)
        
        lasso_data_u_b_l_kk_3773_BW_BWb = self.lasso_calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.truth_phi_kk,self.truth_b123_kk,self.truth_u_b_l_kk_3773)
        
        sum_frac = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3773_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3773_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3773_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3773_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3773_BW_BWb))) 

        sum_strong = np.sum(dplex.dabs(np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_flatte980)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_flatte1270)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_long)+np.einsum("mljk->mjk", lasso_data_phif0_kk_3686_BW_BW_f0_kk)+np.einsum("mljk->mjk", lasso_data_phif2_kk_3686_BW_BW_f2_kk)+np.einsum("mljk->mjk", lasso_data_u_rho_r_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_rho_l_kk_3686_BW_BW)+np.einsum("mljk->mjk", lasso_data_u_b_r_kk_3686_BW_BWb)+np.einsum("mljk->mjk", lasso_data_u_b_l_kk_3686_BW_BWb)))
        total_ff = 1.589
        step_function = (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_flatte980))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_flatte1270))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_long))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_f0_kk))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW_f2_kk))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3686_BW_BW+lasso_data_u_rho_r_kk_3686_BW_BW))/sum_strong)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3686_BW_BWb+lasso_data_u_b_r_kk_3686_BW_BWb))/sum_strong)-1.195,2.0))*10000.0  + 0.0+(np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3773_BW_BW+lasso_data_u_rho_r_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3773_BW_BWb+lasso_data_u_b_r_kk_3773_BW_BWb))/sum_frac)-0.188,2.0))*10000.0 + (np.power(np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3686_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3686_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3686_BW_BW+lasso_data_u_rho_r_kk_3686_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3686_BW_BWb+lasso_data_u_b_r_kk_3686_BW_BWb))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_flatte980))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_flatte1270))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_long))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif0_kk_3773_BW_BW_f0_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_phif2_kk_3773_BW_BW_f2_kk))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_rho_l_kk_3773_BW_BW+lasso_data_u_rho_r_kk_3773_BW_BW))/sum_frac)+np.sum(np.einsum("ljk->l",dplex.dabs(lasso_data_u_b_l_kk_3773_BW_BWb+lasso_data_u_b_r_kk_3773_BW_BWb))/sum_frac)-total_ff,2.0))*10000.0
        # jax_exper.id_print(step_function)
        
        return -np.sum(self.wt_data_kk*np.log(np.sum(dplex.dabs(data_phif0_kk_3686_BW_flatte980 + data_phif2_kk_3686_BW_flatte1270 + data_phif2_kk_3686_BW_BW + data_phif0_kk_3686_BW_BW + data_phif0_kk_3686_BW_BW_long + data_phif0_kk_3686_BW_BW_f0_kk + data_phif2_kk_3686_BW_BW_f2_kk + data_u_rho_r_kk_3686_BW_BW + data_u_rho_l_kk_3686_BW_BW + data_u_b_r_kk_3686_BW_BWb + data_u_b_l_kk_3686_BW_BWb + data_phif0_kk_3773_BW_flatte980 + data_phif2_kk_3773_BW_flatte1270 + data_phif2_kk_3773_BW_BW + data_phif0_kk_3773_BW_BW + data_phif0_kk_3773_BW_BW_long + data_phif0_kk_3773_BW_BW_f0_kk + data_phif2_kk_3773_BW_BW_f2_kk + data_u_rho_r_kk_3773_BW_BW + data_u_rho_l_kk_3773_BW_BW + data_u_b_r_kk_3773_BW_BWb + data_u_b_l_kk_3773_BW_BWb),axis=1))) + step_function
        
    

    
    
    def mc_likelihood_kk(self, args):
        
        kk_3686_amp_inta = np.array([-1.0])
        
        kk_3686_tha_inta = np.array([0.0])
        
        phi_mass = np.array([1.02])
        
        phi_width = np.array([0.004])
        
        kk_f980_mass = np.array([0.9672168652830319])
        
        kk_f980_g_kk = np.array([0.1253740669877132])
        
        kk_f980_rg = np.array([2.7005052659229323])
        
        kk_f980_const_3686 = np.array([0.1,args[0]]).reshape(-1,2)
        
        kk_f980_theta_3686 = np.array([0.1,args[1]]).reshape(-1,2)
        
        kk_f1270_mass = np.array([1.278468123947484])
        
        kk_f1270_width = np.array([0.18253178603942197])
        
        kk_f1270_const_3686 = np.array([args[2],args[3],args[4],args[5],args[6]]).reshape(-1,5)
        
        kk_f1270_theta_3686 = np.array([args[7],args[8],args[9],args[10],args[11]]).reshape(-1,5)
        
        kk_f2_mass = np.array([1.5150133016307599])
        
        kk_f2_width = np.array([0.09287088061436292])
        
        kk_f2_const_3686 = np.array([args[12],args[13],args[14],args[15],args[16]]).reshape(-1,5)
        
        kk_f2_theta_3686 = np.array([args[17],args[18],args[19],args[20],args[21]]).reshape(-1,5)
        
        kk_f0_mass = np.array([2.4753883485342745,2.3212816589917145,1.6998114389127226])
        
        kk_f0_width = np.array([0.09005646310357743,0.08237684207631378,0.15477458920519632])
        
        kk_f0_const_3686 = np.array([args[22],args[23],args[24],args[25],args[26],args[27]]).reshape(-1,2)
        
        kk_f0_theta_3686 = np.array([args[28],args[29],args[30],args[31],args[32],args[33]]).reshape(-1,2)
        
        kk_f0_long_mass = np.array([1.0])
        
        kk_f0_long_width = np.array([100000.0])
        
        kk_f0_long_const_3686 = np.array([args[34],args[35]]).reshape(-1,2)
        
        kk_f0_long_theta_3686 = np.array([args[36],args[37]]).reshape(-1,2)
        
        kk_f0l_mass = np.array([2.064921722894072])
        
        kk_f0l_width = np.array([0.30238434570737827])
        
        kk_f0l_const_3686 = np.array([args[38],args[39]]).reshape(-1,2)
        
        kk_f0l_theta_3686 = np.array([args[40],args[41]]).reshape(-1,2)
        
        kk_f2l_mass = np.array([1.9435205040750276,2.1783628905892916,2.3736695101913585])
        
        kk_f2l_width = np.array([0.46553324520367734,0.171365802944736,0.35232188312114704])
        
        kk_f2l_const_3686 = np.array([args[42],args[43],args[44],args[45],args[46],args[47],args[48],args[49],args[50],args[51],args[52],args[53],args[54],args[55],args[56]]).reshape(-1,5)
        
        kk_f2l_theta_3686 = np.array([args[57],args[58],args[59],args[60],args[61],args[62],args[63],args[64],args[65],args[66],args[67],args[68],args[69],args[70],args[71]]).reshape(-1,5)
        
        kk_u_rho_mass = np.array([2.397312161851163])
        
        kk_u_rho_width = np.array([0.08930983775105905])
        
        kk_u_rho_const_3686 = np.array([args[72]]).reshape(-1,1)
        
        kk_u_rho_theta_3686 = np.array([args[73]]).reshape(-1,1)
        
        kk_u_b_mass = np.array([1.8383688963975429,2.21695086990315])
        
        kk_u_b_width = np.array([0.2886776944601449,0.30820571811928366])
        
        kk_u_b_const_3686 = np.array([args[74],args[75],args[76],args[77],args[78],args[79],args[80],args[81]]).reshape(-1,4)
        
        kk_u_b_theta_3686 = np.array([args[82],args[83],args[84],args[85],args[86],args[87],args[88],args[89]]).reshape(-1,4)
        
        kk_3773_amp_inta = np.array([args[90]])
        
        kk_3773_tha_inta = np.array([args[91]])
        
        kk_f980_const_3773 = np.array([0.1,0.0812758802808357]).reshape(-1,2)
        
        kk_f980_theta_3773 = np.array([0.1,0.04219055453438748]).reshape(-1,2)
        
        kk_f1270_const_3773 = np.array([-0.001908795026008432,-0.002198862452744889,9.263590492855337e-05,0.0038573660399420315,0.004216085267475821]).reshape(-1,5)
        
        kk_f1270_theta_3773 = np.array([0.01328748133437328,0.019267847356889318,0.00823480715160893,-0.0017014797381603949,0.013688984939122168]).reshape(-1,5)
        
        kk_f2_const_3773 = np.array([0.00836160494789744,0.0022731471899687674,-0.0046768086923496405,-0.030976560693756445,0.011016487927278483]).reshape(-1,5)
        
        kk_f2_theta_3773 = np.array([0.0007651896888136746,-0.0021162876037446768,-0.005601756967141078,-0.0013397178875563952,-0.00916615679988274]).reshape(-1,5)
        
        kk_f0_const_3773 = np.array([0.006776209842051121,-0.005211775708005117,0.013982507928474695,0.007221463747811338,0.016393747705371626,0.0026018914837070166]).reshape(-1,2)
        
        kk_f0_theta_3773 = np.array([-0.019419455060917547,-0.02162941877883244,-0.016406491062264772,-0.011508544965435973,-0.02996999596857378,-0.016574134291907173]).reshape(-1,2)
        
        kk_f0_long_const_3773 = np.array([0.17112767897801573,0.05688549960596512]).reshape(-1,2)
        
        kk_f0_long_theta_3773 = np.array([-0.11230387384356917,-0.06356583331362733]).reshape(-1,2)
        
        kk_f0l_const_3773 = np.array([-0.01907412396118519,-0.014324317992303416]).reshape(-1,2)
        
        kk_f0l_theta_3773 = np.array([-0.026948437967964086,-0.009458867738581251]).reshape(-1,2)
        
        kk_f2l_const_3773 = np.array([-0.009557649920876458,-0.0031727092103564524,0.014100451734034392,0.010479172150098524,0.0003692045644304492,0.0018597973868881273,0.00457547644999926,0.003597272060697062,0.0062908372146082804,0.0035507238700523296,-0.017485005393528725,0.015226995460400562,0.0013420411659886469,0.03677091206682049,-0.0004435335025267973]).reshape(-1,5)
        
        kk_f2l_theta_3773 = np.array([-0.017692773889318524,-0.014879688649882716,0.00157904926827019,0.00635591994629108,0.006381438521152356,-0.01207283493112012,-0.0029108265764609606,0.0016613195856758618,0.011585574654273742,0.005166091307635481,-0.0645311336569186,0.003900999450326149,-0.006707506973748528,0.044428897338296275,0.005620130634310312]).reshape(-1,5)
        
        kk_u_rho_const_3773 = np.array([-0.0017169985093901598]).reshape(-1,1)
        
        kk_u_rho_theta_3773 = np.array([-0.003072954777635661]).reshape(-1,1)
        
        kk_u_b_const_3773 = np.array([0.003136380638161863,0.0020679454615716384,-0.0024864348773066037,0.005679669182781839,0.007687456697620569,-0.0014652125102298027,0.004360881343635086,0.004918464768283009]).reshape(-1,4)
        
        kk_u_b_theta_3773 = np.array([-0.00832920259713428,-0.001168027729720911,-0.011101976414934987,-0.003775730069307858,0.009518783705592684,0.011551460555613131,-0.0002691609960762099,0.0002696843264585296]).reshape(-1,4)
        
        
        mc_phif0_kk_3686_BW_flatte980 = self.calculate_BW_flatte980(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3686)
        
        mc_phif2_kk_3686_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3686)
        
        mc_phif2_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3686)
        
        mc_phif0_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3686,kk_f0_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3686)
        
        mc_phif0_kk_3686_BW_BW_long = self.calculate_BW_BW_long(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3686)
        
        mc_phif0_kk_3686_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3686)
        
        mc_phif2_kk_3686_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3686)
        
        mc_u_rho_r_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.mc_phi_kk,self.mc_b124_kk,self.mc_u_rho_r_kk_3686)
        
        mc_u_rho_l_kk_3686_BW_BW = self.calculate_BW_BW(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3686,kk_u_rho_theta_3686,self.mc_phi_kk,self.mc_b123_kk,self.mc_u_rho_l_kk_3686)
        
        mc_u_b_r_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.mc_phi_kk,self.mc_b124_kk,self.mc_u_b_r_kk_3686)
        
        mc_u_b_l_kk_3686_BW_BWb = self.calculate_BW_BWb(kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,self.mc_phi_kk,self.mc_b123_kk,self.mc_u_b_l_kk_3686)
        
        mc_phif0_kk_3773_BW_flatte980 = self.calculate_BW_flatte980(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3773,kk_f980_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3773)
        
        mc_phif2_kk_3773_BW_flatte1270 = self.calculate_BW_flatte1270(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3773,kk_f1270_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3773)
        
        mc_phif2_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3773,kk_f2_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3773)
        
        mc_phif0_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_mass,kk_f0_width,kk_f0_const_3773,kk_f0_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3773)
        
        mc_phif0_kk_3773_BW_BW_long = self.calculate_BW_BW_long(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3773,kk_f0_long_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3773)
        
        mc_phif0_kk_3773_BW_BW_f0_kk = self.calculate_BW_BW_f0_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3773,kk_f0l_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif0_kk_3773)
        
        mc_phif2_kk_3773_BW_BW_f2_kk = self.calculate_BW_BW_f2_kk(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3773,kk_f2l_theta_3773,self.mc_phi_kk,self.mc_f_kk,self.mc_phif2_kk_3773)
        
        mc_u_rho_r_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.mc_phi_kk,self.mc_b124_kk,self.mc_u_rho_r_kk_3773)
        
        mc_u_rho_l_kk_3773_BW_BW = self.calculate_BW_BW(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_rho_mass,kk_u_rho_width,kk_u_rho_const_3773,kk_u_rho_theta_3773,self.mc_phi_kk,self.mc_b123_kk,self.mc_u_rho_l_kk_3773)
        
        mc_u_b_r_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.mc_phi_kk,self.mc_b124_kk,self.mc_u_b_r_kk_3773)
        
        mc_u_b_l_kk_3773_BW_BWb = self.calculate_BW_BWb(kk_3773_amp_inta,kk_3773_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3773,kk_u_b_theta_3773,self.mc_phi_kk,self.mc_b123_kk,self.mc_u_b_l_kk_3773)
        
        return np.sum(dplex.dabs(mc_phif0_kk_3686_BW_flatte980 + mc_phif2_kk_3686_BW_flatte1270 + mc_phif2_kk_3686_BW_BW + mc_phif0_kk_3686_BW_BW + mc_phif0_kk_3686_BW_BW_long + mc_phif0_kk_3686_BW_BW_f0_kk + mc_phif2_kk_3686_BW_BW_f2_kk + mc_u_rho_r_kk_3686_BW_BW + mc_u_rho_l_kk_3686_BW_BW + mc_u_b_r_kk_3686_BW_BWb + mc_u_b_l_kk_3686_BW_BWb + mc_phif0_kk_3773_BW_flatte980 + mc_phif2_kk_3773_BW_flatte1270 + mc_phif2_kk_3773_BW_BW + mc_phif0_kk_3773_BW_BW + mc_phif0_kk_3773_BW_BW_long + mc_phif0_kk_3773_BW_BW_f0_kk + mc_phif2_kk_3773_BW_BW_f2_kk + mc_u_rho_r_kk_3773_BW_BW + mc_u_rho_l_kk_3773_BW_BW + mc_u_b_r_kk_3773_BW_BWb + mc_u_b_l_kk_3773_BW_BWb))
    



    
    
    
    def calculate_BW_flatte980(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f980_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const_3686, kk_f980_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_flatte980(self, phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.flatte980,Sbc=f_kk))(kk_f980_mass,kk_f980_g_kk,kk_f980_rg),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_flatte1270(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f1270_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const_3686, kk_f1270_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_flatte1270(self, phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.flatte1270,Sbc=f_kk))(kk_f1270_mass,kk_f1270_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2_const_3686, kk_f2_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW(self, phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW,Sbc=f_kk))(kk_f2_mass,kk_f2_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_long(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0_long_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_long(phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_long_const_3686, kk_f0_long_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_long_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_long(self, phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_long,Sbc=f_kk))(kk_f0_long_mass,kk_f0_long_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_f0_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f0_kk(phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0l_const_3686, kk_f0l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_f0_kk(self, phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f0_kk,Sbc=f_kk))(kk_f0l_mass,kk_f0l_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BW_f2_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f2_kk(phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2l_const_3686, kk_f2l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # print("inta_ph",inta_ph.shape)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)#给每个相位因子进行相位修正
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)#给共振态的每个振幅乘上相位因子并把该共振态的全部振幅求和
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->jk", phif, bw)
        
        return phif
    def BW_BW_f2_kk(self, phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        # a = self.BW_phi(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f2_kk,Sbc=f_kk))(kk_f2l_mass,kk_f2l_width),1,0)

        return dplex.deinsum("j, ij->ij",a,b)

    
    def calculate_BW_BWb(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,phi_kk,b124_kk,u_b_r_kk_3686):

        
        bw1 = self.BW_BW_f0_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)
        bw2 = self.BW_BW_f2_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)

        amp_caled_1 = u_b_r_kk_3686[[0, 2], :, :]  # SS and DS
        amp_caled_2 = u_b_r_kk_3686[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct(kk_u_b_const_3686, kk_u_b_theta_3686)
        
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->jk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        
        return phif 
    def BW_BW_f0_k0(self, phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f0_k0,Sbc=b124_kk))(kk_u_b_mass,kk_u_b_width),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    def BW_BW_f2_k0(self, phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk):
        a = self.BW(phi_mass,phi_width,phi_kk)
        b = np.moveaxis(vmap(partial(self.BW_f2_k0,Sbc=b124_kk))(kk_u_b_mass,kk_u_b_width),1,0)
        return dplex.deinsum("j, ij->ij",a,b)
    
    def lasso_calculate_BW_flatte980(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,kk_f980_const_3686,kk_f980_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f980_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte980(phi_mass,phi_width,phi_kk,kk_f980_mass,kk_f980_g_kk,kk_f980_rg,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f980_const_3686, kk_f980_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f980_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_flatte1270(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f1270_mass,kk_f1270_width,kk_f1270_const_3686,kk_f1270_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f1270_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_flatte1270(phi_mass,phi_width,phi_kk,kk_f1270_mass,kk_f1270_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f1270_const_3686, kk_f1270_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f1270_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2_mass,kk_f2_width,kk_f2_const_3686,kk_f2_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW(phi_mass,phi_width,phi_kk,kk_f2_mass,kk_f2_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2_const_3686, kk_f2_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_long(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0_long_mass,kk_f0_long_width,kk_f0_long_const_3686,kk_f0_long_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0_long_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_long(phi_mass,phi_width,phi_kk,kk_f0_long_mass,kk_f0_long_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0_long_const_3686, kk_f0_long_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0_long_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_f0_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f0l_mass,kk_f0l_width,kk_f0l_const_3686,kk_f0l_theta_3686,phi_kk,f_kk,phif0_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f0l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f0_kk(phi_mass,phi_width,phi_kk,kk_f0l_mass,kk_f0l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif0_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f0l_const_3686, kk_f0l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f0l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif0_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BW_f2_kk(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_f2l_mass,kk_f2l_width,kk_f2l_const_3686,kk_f2l_theta_3686,phi_kk,f_kk,phif2_kk_3686):

        
        # ph = np.moveaxis(self.phase(kk_f2l_theta_3686), 1, 0)
        # print("phif",phif.shape)
        # print("phase",ph.shape)
        bw = self.BW_BW_f2_kk(phi_mass,phi_width,phi_kk,kk_f2l_mass,kk_f2l_width,f_kk)
        # print("bw", bw.shape)
        # phif = dplex.dtomine(phif2_kk_3686)
        # print("phif",phif.shape)
        const_ph = dplex.dconstruct(kk_f2l_const_3686, kk_f2l_theta_3686)
        # const_ph = dplex.deinsum_ord("li,li->li", np.exp(kk_f2l_const_3686), ph)
        # print("const_ph",const_ph.shape)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        phif = dplex.deinsum_ord("ijk,li->ljk", phif2_kk_3686, const_ph)
        # phif = dplex.deinsum("ijk,li->ljk", phif, const_ph)
        phif = dplex.deinsum("ljk,lj->ljk", phif, bw)
        # lasso_phif = np.einsum("ljk->l",dplex.dabs(phif))
        # return lasso_phif
        
        return phif
    
    def lasso_calculate_BW_BWb(self,kk_3686_amp_inta,kk_3686_tha_inta,phi_mass,phi_width,kk_u_b_mass,kk_u_b_width,kk_u_b_const_3686,kk_u_b_theta_3686,phi_kk,b124_kk,u_b_r_kk_3686):

        
        bw1 = self.BW_BW_f0_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)
        bw2 = self.BW_BW_f2_k0(phi_mass,phi_width,phi_kk,kk_u_b_mass,kk_u_b_width,b124_kk)

        amp_caled_1 = u_b_r_kk_3686[[0, 2], :, :]  # SS and DS
        amp_caled_2 = u_b_r_kk_3686[[1, 3], :, :]  # DS and DD

        const_ph = dplex.dconstruct(kk_u_b_const_3686, kk_u_b_theta_3686)
        inta_ph = dplex.dconstruct(kk_3686_amp_inta, kk_3686_tha_inta)
        # const_ph = dplex.deinsum("lij,l->lij", const_ph, inta_ph)
        const_ph = dplex.deinsum("li,l->li", const_ph, inta_ph)
        const_ph1 = const_ph[:, :, [0, 2]]
        const_ph2 = const_ph[:, :, [1, 3]]

        amp_caled_wphase1 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_1, const_ph1)
        amp_caled_wphase2 = dplex.deinsum_ord("ijk,li->ljk", amp_caled_2, const_ph2)

        phif1 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase1, bw1)
        phif2 = dplex.deinsum("ljk,lj->ljk", amp_caled_wphase2, bw2)

        phif = phif1 + phif2
        
        return phif

    def phase(self, theta):
        return vmap(self._phase)(theta)

    def _phase(self, theta):
        return dplex.dconstruct(np.cos(theta), np.sin(theta))
    
    
    def BW(self, m_,w_,Sbc):
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(1.0, temp)

    def BW_relativity(self, m_,w_,Sbc):
        gamma=np.sqrt(m_*m_*(m_*m_+w_*w_))
        k = np.sqrt(2*np.sqrt(2)*m_*np.abs(w_)*gamma/np.pi/np.sqrt(m_*m_+gamma))
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(m_*m_ - Sbc,  -m_*w_*np.ones(l))
        return dplex.ddivide(k, temp)

    def BW_phi(self,m_,w_,Sbc):
        l = (Sbc.shape)[0]
        m_k = 0.493677
        q_0 = 1/3
        q_m = np.sqrt(Sbc-4*m_k*m_k)/2
        q_phi = np.sqrt(m_*m_-4*m_k*m_k)/2
        b_qm = 1 / np.sqrt(1+(q_m/q_0)*(q_m/q_0))
        b_qphi = 1 / np.sqrt(1+(q_phi/q_0)*(q_phi/q_0))

        k = q_m / q_0 *b_qm
        Gamma_phi = w_ * m_ / np.sqrt(Sbc)*(q_m/q_phi)*(q_m/q_phi)*b_qm*b_qm/b_qphi/b_qphi

        temp = dplex.dconstruct(m_*m_ - Sbc, -m_*Gamma_phi*np.ones(l))
        return dplex.ddivide(k, temp)

    def flatte980(self,m_,g_pipi,rg,Sbc):
        # 对于phipipi的过程的rho_kk的计算，不能单纯取绝对值，因为有一部分pipi在kk的质量阈值之下
        g_kk = rg * g_pipi
        m_k = 0.493677
        m_pi = 0.13957061

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        sign_kk = np.sign(tmp_kk)
        tmp_pipi = 1 - 4*m_pi*m_pi / Sbc
        sign_pipi = np.sign(tmp_pipi)

        # 若这一项大于零，应该留在虚部，如果小于零，应该*加*到实部

        real_kk_factor = ( tmp_kk - sign_kk * tmp_kk ) / (tmp_kk * 2)
        img_kk_factor = ( tmp_kk + sign_kk * tmp_kk ) / (tmp_kk * 2)

        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        tmp_A = dplex.dconstruct(m_**2 - Sbc + g_kk*rho_kk*real_kk_factor, -1*m_*(g_pipi*rho_pipi + g_kk*rho_kk*img_kk_factor))
        return dplex.ddivide(1.0, tmp_A)

    def flatte1270(self,m_,w_,Sbc):
        rm = m_ * m_
        gr = m_ * w_
        q2r = 0.25 * rm - 0.0194792#这实际上是一个参数化宽度的BW(PBW)，其中scale factor这里取1fm
        b2r = q2r * (q2r + 0.1825) + 0.033306
        g11270 = gr * b2r / np.power(q2r,2.5)
        q2 = 0.25 * Sbc - 0.0194792
        b2 = q2 * (q2 + 0.1825) + 0.033306
        g1 = g11270 * np.power(q2,2.5) / b2
        tmp = dplex.dconstruct(Sbc - rm, g1)
        return dplex.ddivide(gr, tmp)
    
    def flatte500(self,m_,b1,b2,A,Sbc):
        m_pi = 0.139556995
        m_k = 0.493677
        m_eta = 0.547862
        S_a = 0.41*m_pi*m_pi
        m2 = m_*m_
        alpha = 1.3
        g4 = 0.011 # 参考文献里有很多取值

        g1_sqr = m_*(b1+b2*Sbc)*np.exp(-(Sbc-m_*m_)/A)
        rho_pipi = np.sqrt(np.abs(1 - 4*m_pi*m_pi / Sbc))
        rho_M = np.sqrt(np.abs(1 - 4*m_pi*m_pi / (m_*m_)))

        tmp_kk = 1 - 4*m_k*m_k / Sbc
        rho_kk = np.sqrt(np.abs(1 - 4*m_k*m_k / Sbc))
        tmp_nn = 1 - 4*m_eta*m_eta / Sbc
        rho_nn = np.sqrt(np.abs(1 - 4*m_eta*m_eta / Sbc))

        rho_kk_sign = np.sign(1 - 4*m_k*m_k / Sbc) 
        rho_nn_sign = np.sign(1 - 4*m_eta*m_eta / Sbc)

        kk_real_factor = ( tmp_kk - rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        kk_img_factor = ( tmp_kk + rho_kk_sign * tmp_kk ) / (tmp_kk * 2)
        nn_real_factor = ( tmp_nn - rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        nn_img_factor = ( tmp_nn + rho_nn_sign * tmp_nn ) / (tmp_nn * 2)
        # pipi 不变质量会出现在 kk 和 etaeta 质量阈值 之下
        
        rho_4pis = 1.0/(1+np.exp(7.082-2.845*Sbc))
        rho_4pim = 1.0/(1+np.exp(7.082-2.845*(m_*m_)))

        mgam1 = g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*rho_pipi
        mgam2 = 0.6*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_k*m_k))*rho_kk 
        mgam3 = 0.2*g1_sqr*(Sbc/m2)*np.exp(-1*alpha*np.abs(Sbc-4*m_eta*m_eta))*rho_nn 
        mgam4 = m_*g4*rho_4pis/rho_4pim

        j1_s = (2 + rho_pipi * np.log((1-rho_pipi)/(1+rho_pipi))) / np.pi #log本身就是以e为底
        j1_m = (2 + rho_M * np.log((1-rho_M )/(1+rho_M ))) / np.pi
        zs = j1_s - j1_m 

        # mgam = mgam1 + mgam2 + mgam3 + mgam4 
        T_pipi = dplex.dconstruct(m_*m_ - Sbc - g1_sqr*(Sbc-S_a)/(m_*m_-S_a)*zs + kk_real_factor*mgam2 + nn_real_factor*mgam3 ,mgam1+mgam4 + kk_img_factor*mgam2 +nn_img_factor *mgam3 )

        return dplex.ddivide(1.0,T_pipi)

    def BW_long(self, m_,w_,Sbc):
        # LHCb note BW=1 for non resonnace
        l = (Sbc.shape)[0]
        temp = dplex.dconstruct(np.ones(l),  np.zeros(l))
        return dplex.ddivide(1.0, temp)

    def BW_f0_kk(self,m_,w_,Sbc):# paramtered width l=0
        S_b = 0.24371698
        S_c = 0.24371698
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0

        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f2_kk(self,m_,w_,Sbc):# paramtered width l=2
        S_b = 0.24371698
        S_c = 0.24371698
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)

        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f0_pipi(self,m_,w_,Sbc):# paramtered width l=0
        S_b = 0.0194797849
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0

        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_f2_pipi(self,m_,w_,Sbc):# paramtered width l=2
        S_b = 0.0194797849
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)

        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)

    def BW_rho770(self,m_,w_,Sbc):# GS_lineshape
        S_b = 0.0194797849
        S_c = 0.0194797849
        GS1 = 0.636619783 # 2/ma.pi
        GS2 = 0.01860182466 # 3*mpion2/ma.pi
        GS3 = 0.1591549458; # 1/(2*ma.pi)
        GS4 = 0.00620060822; # mass_Pion2/ma.pi
        m = np.sqrt(Sbc)
        m02 = m_*m_ # m0 = m_ 是我们要拟合的参数
        q2 = 0.25*(Sbc + S_b - S_c)**2/Sbc - S_b
        q = np.sqrt(q2)
        q02 = 0.25*(m02 + S_b - S_c)**2/m02 - S_b
        q0 = np.sqrt(q02)
        q03 = q0*q02
        t = q/q0
        #按照zbs的文章和程序其他的部分，R被设置为1
        bf = (1+q02)/(1+q2)# rho2pipi l=1
        #log(mass_2Pion) = 1.2760418309
        h  = GS1*q/m*(np.log(m+2*q)+1.2760418309)
        h0 = GS1*q0/m_*(np.log(m_+2*q0)+1.2760418309)
        dh = h0*(0.125/q02-0.5/m_)+GS3/m02
        d  = GS2/q02*(np.log(m_+2*q0)+1.2760418309)+GS3*m_/q0-GS4*m_/q03

        gam = w_*t*t*t*m_/m*bf 
        f = w_*m02/q03*(q2*(h-h0)+(m02-Sbc)*q02*dh)

        a = 1.0 + d*w_/m_
        b = dplex.dconstruct(m_*m_ - Sbc + f, -m_*gam)
        return dplex.ddivide(a, b)
    def BW_f0_b1(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.0392986916 #b1=m_phi+m_pi
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0
        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f2_b1(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.0392986916 #b1=m_phi+m_pi
        S_c = 0.0194797849
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)
        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f0_k0(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.2627 #kst=m_phi+m_k
        S_c = 0.24372
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = q/q0
        gam = w_*t*m_/Sbc # barrier_factor == 1
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
    def BW_f2_k0(self,m_,w_,Sbc):# paramtered width l=0 b1 state SS and DS
        S_b = 1.2627 #kst=m_phi+m_k
        S_c = 0.24372
        m02 = m_*m_
        q2 = np.abs(0.25*(Sbc + S_b - S_c)**2/Sbc - S_b)
        q = np.sqrt(q2)
        q02 = np.abs(0.25*(m02 + S_b - S_c)**2/m02 - S_b)
        q0 = np.sqrt(q02)
        t = (q/q0)**5#not sure who is more accuracy
        bf = (9+3*q02+q02*q02)/(9+3*q2+q2*q2)
        gam = w_*t*m_/Sbc*bf
        tmp = dplex.dconstruct(m_*m_ - Sbc,-m_*gam)
        return dplex.ddivide(1.0, tmp)
# 可以发现，传播子在结构上相似，对于同一个轨道角动量的情况差距仅出现在Sb和Sc上，
# 因此最终对于相同l的结果可以统一为一个函数，只需要传入不同的Sb和Sc即可
# 最终的函数应该可以构造为 BW_f0(self,m_,w_,Sbc,Sb,Sc)




    
    def hvp_data_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.data_likelihood_kk), [args_float], [any_vector])

    def hvp_mc_fwdrev_kk(self, args_float, any_vector):
        return jvp(grad(self.mc_likelihood_kk), [args_float], [any_vector])
    

    def jit_request(self):
        
        self.jit_data_likelihood_kk = jit(self.data_likelihood_kk, device=self.device)
        self.jit_mc_likelihood_kk = jit(self.mc_likelihood_kk, device=self.device)
        self.jit_grad_data_likelihood_kk = jit(grad(self.data_likelihood_kk), device=self.device)
        self.jit_grad_mc_likelihood_kk = jit(value_and_grad(self.mc_likelihood_kk), device=self.device)
        self.jit_hvp_data_fwdrev_kk = jit(self.hvp_data_fwdrev_kk, device=self.device)
        self.jit_hvp_mc_fwdrev_kk = jit(self.hvp_mc_fwdrev_kk, device=self.device)
        

    




class Control(object):
    def __init__(self, args):
        self.bic_delete_mods = args.bic_delete_mods
        self.total_gpu_id = args.total_gpu_id
        self.processes_gpus = args.processes_gpus
        self.max_processes = args.max_processes
        self.max_processes_memory = args.max_processes_memory
        self.thread_gpus = args.thread_gpus
        self.threads_in_one_gpu = args.threads_in_one_gpu
        
        self.data_config = dict()
        self.data_config["data_slices"] = args.data_slices
        self.data_config["mc_slices"] = args.mc_slices
        self.data_config["mini_run"] = args.mini_run
        
        self.args_list = onp.array([-1.0, 0.0, 1.02, 0.004, 0.9672168652830319, 0.1253740669877132, 2.7005052659229323, 0.1, 0.02528579544349515, 0.1, 0.08996768704897123, 1.278468123947484, 0.18253178603942197, -0.00394557414604787, -0.031456917490503056, -0.006453975683670302, -0.000751259137593863, 0.005700617518362637, 0.0019064455025013442, -0.016674047180805302, 0.00042242238223968634, -0.03844830111424663, 0.005482331962888952, 1.5150133016307599, 0.09287088061436292, 0.013989586500962098, 6.536487778784994e-05, -0.003039878487010079, -0.02290051721070652, -0.005149185577791717, -0.002271692391140229, 0.0014266102701488276, -0.0002464838712524822, 0.027487458758986043, -0.021217704602433694, 2.4753883485342745, 2.3212816589917145, 1.6998114389127226, 0.09005646310357743, 0.08237684207631378, 0.15477458920519632, 0.020345057172526142, -0.019615646872485935, -0.011694357209471467, -0.011750000184762972, 0.01586061786100111, -0.012419424802564446, -0.044498982664504055, -0.04475789168359524, -0.007019105704104814, -0.010339726069162758, 0.023800660913466182, -0.009236629506304373, 1.0, 100000.0, 0.018412616428206293, -0.014362887948466602, 0.010468017555146498, 0.08366133891030483, 2.064921722894072, 0.30238434570737827, -0.008061498374653295, -0.009561621302788686, 0.025185456013695852, 0.020110419942420712, 1.9435205040750276, 2.1783628905892916, 2.3736695101913585, 0.46553324520367734, 0.171365802944736, 0.35232188312114704, 0.010712196314346355, 0.00347347287437411, -0.001123202599978466, 0.003038246648574247, 0.006169230483688115, -0.008715046306530293, -0.0037697755182356366, -0.0021439357115616313, 0.009843889611037352, -0.002311207291324084, -0.027166615697145177, -0.03686780441569233, -0.030199614610973834, -0.014486397366395492, -0.03600045818702371, 0.019906238124680783, -0.02259436401747235, 0.004982846050260414, -0.03157580467041962, -0.0029303473657870892, 0.011827388845592392, 0.003308659428709971, 0.0031953094204852034, -0.00042027920310028563, 0.007031918963374428, -0.0608692440971674, -0.011852304444291448, -0.004489335843117047, 0.044159240796422666, 0.014583440328481448, 2.397312161851163, 0.08930983775105905, 0.0022778257328804684, -0.0033688144994624505, 1.8383688963975429, 2.21695086990315, 0.2886776944601449, 0.30820571811928366, -0.004938490390296392, 0.016593974473936707, 0.003986752368883177, 0.012164816898769098, 0.005186646362675694, 0.0029039740223936127, 0.007183982423137571, 0.007637037419706918, -0.0016244356428979153, 0.0010019867956887615, 0.004619734538692254, 0.01106119138947309, 0.012480170130002325, 0.0025652685920836142, 0.0005478969786869395, 0.0012203694483755145, 0.0, 0.2, 0.1, 0.0812758802808357, 0.1, 0.04219055453438748, -0.001908795026008432, -0.002198862452744889, 9.263590492855337e-05, 0.0038573660399420315, 0.004216085267475821, 0.01328748133437328, 0.019267847356889318, 0.00823480715160893, -0.0017014797381603949, 0.013688984939122168, 0.00836160494789744, 0.0022731471899687674, -0.0046768086923496405, -0.030976560693756445, 0.011016487927278483, 0.0007651896888136746, -0.0021162876037446768, -0.005601756967141078, -0.0013397178875563952, -0.00916615679988274, 0.006776209842051121, -0.005211775708005117, 0.013982507928474695, 0.007221463747811338, 0.016393747705371626, 0.0026018914837070166, -0.019419455060917547, -0.02162941877883244, -0.016406491062264772, -0.011508544965435973, -0.02996999596857378, -0.016574134291907173, 0.17112767897801573, 0.05688549960596512, -0.11230387384356917, -0.06356583331362733, -0.01907412396118519, -0.014324317992303416, -0.026948437967964086, -0.009458867738581251, -0.009557649920876458, -0.0031727092103564524, 0.014100451734034392, 0.010479172150098524, 0.0003692045644304492, 0.0018597973868881273, 0.00457547644999926, 0.003597272060697062, 0.0062908372146082804, 0.0035507238700523296, -0.017485005393528725, 0.015226995460400562, 0.0013420411659886469, 0.03677091206682049, -0.0004435335025267973, -0.017692773889318524, -0.014879688649882716, 0.00157904926827019, 0.00635591994629108, 0.006381438521152356, -0.01207283493112012, -0.0029108265764609606, 0.0016613195856758618, 0.011585574654273742, 0.005166091307635481, -0.0645311336569186, 0.003900999450326149, -0.006707506973748528, 0.044428897338296275, 0.005620130634310312, -0.0017169985093901598, -0.003072954777635661, 0.003136380638161863, 0.0020679454615716384, -0.0024864348773066037, 0.005679669182781839, 0.007687456697620569, -0.0014652125102298027, 0.004360881343635086, 0.004918464768283009, -0.00832920259713428, -0.001168027729720911, -0.011101976414934987, -0.003775730069307858, 0.009518783705592684, 0.011551460555613131, -0.0002691609960762099, 0.0002696843264585296])
        self.float_list = onp.array([8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 58, 61, 62, 63, 64, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 103, 104, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126])

        max_processes = self.max_processes
        self.process_pool = [Process()] * max_processes
        self.kk_data_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        self.kk_mc_lh = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.kk_data_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_grad = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_grad_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])

        self.kk_data_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp_g = onp.zeros([self.thread_gpus,self.threads_in_one_gpu,self.float_list.shape[0]])
        self.kk_mc_hvp_v = onp.zeros([self.thread_gpus,self.threads_in_one_gpu])
        

    def grad_double(self,a,b,x):
        return (b-a)/2.0*onp.cos(x)
    
    def grad_a(self,x):
        return x/onp.sqrt(onp.power(x,2)+1)

    def grad_b(self,x):
        return -1.0*x/onp.sqrt(onp.power(x,2)+1)

    def likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_data_likelihood_kk(args_float)   
            self.kk_mc_lh[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)   
        

    def thread_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        
        # result_kk = onp.sum(self.kk_data_lh) + 21181.60837312616*onp.log(onp.sum(self.kk_mc_lh))
        result_kk = onp.sum(self.kk_data_lh) + 21181.60837312616*onp.log(onp.sum(self.kk_mc_lh)/925000.0)
        
        result =+ result_kk
        return result

    def jit_grad_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_data_likelihood_kk(args_float)
            self.kk_mc_grad_v[device_rank,_], self.kk_mc_grad[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_grad_mc_likelihood_kk(args_float)
        

    def thread_grad_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_grad_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        
        result_kk = onp.sum(self.kk_data_grad,axis=(0,1)) + 21181.60837312616*onp.sum(self.kk_mc_grad,axis=(0,1))/onp.sum(self.kk_mc_grad_v)
        
        result =+ result_kk
        return result

    def mc_likelihood_in_sigle_device(self, args_float, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_mc_hvp_v[device_rank,_] = self.pwaf_list[device_rank][_].jit_mc_likelihood_kk(args_float)
        

    def thread_mc_likelihood(self, args_float):
        threading_list = [ None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.mc_likelihood_in_sigle_device, args = (args_float, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        
        self.kk_mc_hvp_value = onp.sum(self.kk_mc_hvp_v)
        

    def jit_hvp_in_sigle_device(self, args_float, any_vector, device_rank):
        for _ in range(self.threads_in_one_gpu):
        
            self.kk_data_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_data_fwdrev_kk(args_float, any_vector)[1]
            self.kk_mc_hvp_g[device_rank,_,:], self.kk_mc_hvp[device_rank,_,:] = self.pwaf_list[device_rank][_].jit_hvp_mc_fwdrev_kk(args_float, any_vector)
        

    def thread_hvp(self, args_float, any_vector):
        threading_list = [ None for _ in range(self.thread_gpus)]
        result = [None for _ in range(self.thread_gpus)]
        for _ in range(self.thread_gpus):
            threading_list[_] = Thread(target = self.jit_hvp_in_sigle_device, args = (args_float, any_vector, _))
            threading_list[_].daemon = 1
        for _ in range(self.thread_gpus):
             threading_list[_].start()
        for _ in range(self.thread_gpus):
             threading_list[_].join()
        self.thread_mc_likelihood(args_float)
        
        result_kk = onp.sum(self.kk_data_hvp, axis=(0,1)) + 21181.60837312616*(onp.sum(self.kk_mc_hvp,axis=(0,1))/self.kk_mc_hvp_value - onp.einsum("ikm,jln,n->m",self.kk_mc_hvp_g,self.kk_mc_hvp_g,any_vector)/(self.kk_mc_hvp_value**2))
        
        result =+ result_kk
        return result
    
    def require_compile(self, fcn, *args):
        logger.info("{} compile start".format(fcn.__name__))
        t1 = time.time()
        fcn(*args)
        t2 = time.time()
        logger.info("{} compile complete, time is {}".format(fcn.__name__,t2-t1))

    def my_callback(self, xk):
        logger.info(" fcn: {}".format(self.thread_likelihood(xk)))

    def compile_func(self):
        start = time.time()
        vector = onp.ones(self.args_float.shape)
        compile_likelihood = Thread(target = self.require_compile, args = (self.thread_likelihood, self.args_float))
        compile_grad = Thread(target = self.require_compile, args = (self.thread_grad_likelihood, self.args_float))
        compile_hvp = Thread(target = self.require_compile, args = (self.thread_hvp, self.args_float, vector))

        compile_hvp.start()
        compile_likelihood.start()
        compile_grad.start()

        compile_hvp.join()
        compile_likelihood.join()
        compile_grad.join()
        stop = time.time()
        logger.info("compile time: {}".format(stop-start))

    def run(self, process_initializer, process_returns):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(lambda x:str(x),process_initializer[0][0].total_gpu_id))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=str(process_initializer[0][0].gpu_memory_limit_percentage)
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false" # 用多少拿多少的选项
        config.update("jax_enable_x64", True)

        logger.info("fit_pull_sample begins!!!!")

        start_time = time.time()

        device_list = jdevices()
        self.pwaf_list = [[PWAFunc(process_initializer[i][j], device_list[process_initializer[i][j].gpu_id]) for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
        for i in range(self.thread_gpus):
            for j in range(self.threads_in_one_gpu):
                self.pwaf_list[i][j].jit_request()
        
        num_seed=onp.random.randint(low=0, high=1000000, size=1, dtype='l')
        logger.info("random seed: {}".format(num_seed))
        onp.random.seed(num_seed)
        self.args_float = self.args_list[self.float_list]
        theta_index = onp.array([1, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 28, 29, 30, 31, 32, 33, 36, 37, 40, 41, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 82, 83, 84, 85, 86, 87, 88, 89])
        const_index = onp.array([0, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 34, 35, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 72, 74, 75, 76, 77, 78, 79, 80, 81])
        mass_index = onp.array([])
        width_index = onp.array([])
        inta_amp_index = onp.array([90])
        inta_tha_index = onp.array([91])
        target_index = onp.concatenate([inta_amp_index, inta_tha_index])
        mw_index = onp.array([])
        mw_range = []

        self.compile_func()
        min_fcn = 40000.0
        result_info = dict()
        for n in range(1):
            
            args_float = self.args_float
            t1 = time.time()
            res = minimize(fun=self.thread_likelihood, x0=args_float, jac=self.thread_grad_likelihood, hessp=self.thread_hvp, method="Newton-CG", callback=self.my_callback, options={"disp":False, "xtol":1e-6})
            t2 = time.time()
            logger.info("{0} {1} {2}".format("="*15,str(n),"="*15))
            logger.info("fcn: {}".format(res.fun))
            logger.info("success: {}".format(res.success))
            logger.info("message: {}".format(res.message))
            logger.info("Iterations: {}".format(res.nit))
            logger.info("Function evaluations: {}".format(res.nfev))
            logger.info("Gradient evaluations: {}".format(res.njev))
            logger.info("Hessian evaluations: {}".format(res.nhev))
            logger.info("time: {}".format(t2-t1))
            logger.info("{0}".format("="*33))
            result_info["result_"+str(n)] = {"fcn":res.fun,"success":res.success,"Iterations":res.nit,"FunEval":res.nfev,"GrdEval":res.njev,"HesEval":res.nhev,"time":t2-t1}
            values = res.x
            logger.info("result likelihood: {}".format(self.thread_likelihood(values)))
            fvalue = copy.deepcopy(self.args_list)
            for i, locate in enumerate(self.float_list):
                fvalue[locate] = values[i]

            
            # if res.fun < min_fcn and res.success:
            if res.fun < min_fcn:
                min_fcn = res.fun

            args_float_size = (args_float.shape)[0]
            my_hessian = onp.zeros([args_float_size,args_float_size])
            for x in range(args_float_size):
                v = onp.zeros(args_float_size)
                v[x] = 1.0
                my_hessian[:,x] = self.thread_hvp(args_float,v)
            # print("eig :",onp.linalg.eig(my_hessian)[0])
            # 以下全为新添加注释
            ferror = onp.sqrt(onp.diag(onp.linalg.inv(my_hessian)))
            fvalue_float = fvalue[self.float_list]
            error = onp.zeros((self.args_list.shape)[0])
            for i, locate in enumerate(self.float_list):
                error[locate] = ferror[i]
            
            logger.info("error: {}".format(error))
            self.save_in_json(fvalue,error,"output/fit/fit_result_kk",str(n),result_info)

            # correlation = onp.linalg.inv(my_hessian)
            # print("correlation :",correlation)
            # onp.save("correlation",correlation)

        min_fcn = [min_fcn]
        logger.info("{0} the minist value {1} {0}".format("*"*4,min_fcn))

        stop_time = time.time()
        logger.info("=================set return=====================")
        process_returns.set("parameter_values", fvalue)
        process_returns.set("parameter_errors", ferror)
        process_returns.set("min_fcn", min_fcn)
        process_returns.info()
        process_returns.set("timer", stop_time - start_time)
        process_returns.set("process_id", os.getpid())
        process_returns.set("gpu_id", process_initializer[0][0].total_gpu_id)

        return

    def run_multiprocess(self):
        self.process_initializer = Process_Initializer_Generator(**self.data_config)
        process_initializer_generator = self.process_initializer.process_initializer_generator()
        self.process_returns = []
        self.start_time = time.time()
        while True:
            try:
                for _proc_numbering, _proc in enumerate(self.process_pool):
                    if not _proc.is_alive():
                        try:
                            _proc.join()
                        except AssertionError:
                            pass
                        finally:
                            # 创建多线程拆分模型, 注意! 一个进程只可以用来计算一个完整的拟合
                            # 线程模型是[i, j]形状, 对应[gpus, thread_in_gpus], 总线程为 gpus*thread_in_gpus
                            _proc_initializer_list = [[None for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)]
                            for i in range(self.thread_gpus):
                                for j in range(self.threads_in_one_gpu):
                                    with next(process_initializer_generator) as _proc_initializer_list[i][j]:
                                        logger.info("data: {}".format(_proc_initializer_list[i][j].data_numbering))
                                        _proc_initializer_list[i][j].gpu_memory_limit_percentage = self.max_processes_memory 
                                        _proc_initializer_list[i][j].gpu_id = i + _proc_numbering % self.processes_gpus # gpu的分配靠 thread_gpus 和 processes_gpus 和进程数调节
                                        _proc_initializer_list[i][j].total_gpu_id = self.total_gpu_id
                            
                            _proc_returns = ProcessReturns()
                            _proc_returns.set("data_numbering", [[_proc_initializer_list[i][j].data_numbering for j in range(self.threads_in_one_gpu)] for i in range(self.thread_gpus)])
                            self.process_pool[_proc_numbering] = Process(target=self.run, args=(_proc_initializer_list, _proc_returns))
                            self.process_pool[_proc_numbering].start()
                            self.process_returns.append(_proc_returns)
                            logger.info("=========== chambering data_numbering.{} ============".format(_proc_returns.get("data_numbering")))

            except StopIteration:
                for _ in self.process_pool:
                    try:
                        _.join()
                    except AssertionError:
                        pass
                break

        self.stop_time = time.time()
        logger.info("Total time: {}".format(self.stop_time - self.start_time))
    
    def get_result_dict(self):
        self.fvalues = (self.process_returns[0]._dict)["parameter_values"]
        self.fcn = (self.process_returns[0]._dict)["min_fcn"]

    def save_in_npz(self):
        logger.info("save in npz")
        total_values = []
        total_errors = []
        for _ in self.process_returns:
            total_values.append(_._dict["parameter_values"])
            total_errors.append(_._dict["parameter_errors"])
        np_values = onp.squeeze(onp.array(total_values), axis=None)
        np_errors = onp.squeeze(onp.array(total_errors), axis=None)
        logger.debug("{}".format(np_values.shape))
        logger.debug("{}".format(np_errors.shape))
        if total_values[0] is not None or total_errors[0] is not None:
            onp.savez("output/fit/fit_result_kk", fvalue=np_values, ferror=np_errors)

        logger.info("program over")
    
    def save_in_json(self, np_values, np_errors, save_addr = "", n = "",result_info={}):
        logger.info("save in json")
        json_addr_list = []
        
        json_addr_list.append("pwa_info_kk.json")
        
        name_list = ['inta_c_3686', 'inta_t_3686', 'phi_m', 'phi_w', 'kk_f980_m', 'kk_g_kk', 'kk_rg', 'kk_f980_c1_3686', 'kk_f980_c2_3686', 'kk_f980_t1_3686', 'kk_f980_t2_3686', 'kk_f1270_m', 'kk_f1270_w', 'kk_f1270_c1_3686', 'kk_f1270_c2_3686', 'kk_f1270_c3_3686', 'kk_f1270_c4_3686', 'kk_f1270_c5_3686', 'kk_f1270_t1_3686', 'kk_f1270_t2_3686', 'kk_f1270_t3_3686', 'kk_f1270_t4_3686', 'kk_f1270_t5_3686', 'kk_f1525_m', 'kk_f1525_w', 'kk_f1525_c1_3686', 'kk_f1525_c2_3686', 'kk_f1525_c3_3686', 'kk_f1525_c4_3686', 'kk_f1525_c5_3686', 'kk_f1525_t1_3686', 'kk_f1525_t2_3686', 'kk_f1525_t3_3686', 'kk_f1525_t4_3686', 'kk_f1525_t5_3686', 'kk_f2470_m', 'kk_f2330_m', 'kk_f1710_m', 'kk_f2470_w', 'kk_f2330_w', 'kk_f1710_w', 'kk_f2470_c1_3686', 'kk_f2470_c2_3686', 'kk_f2330_c1_3686', 'kk_f2330_c2_3686', 'kk_f1710_c1_3686', 'kk_f1710_c2_3686', 'kk_f2470_t1_3686', 'kk_f2470_t2_3686', 'kk_f2330_t1_3686', 'kk_f2330_t2_3686', 'kk_f1710_t1_3686', 'kk_f1710_t2_3686', 'kk_f0long_m', 'kk_f0long_w', 'kk_f0long_c1_3686', 'kk_f0long_c2_3686', 'kk_f0long_t1_3686', 'kk_f0long_t2_3686', 'kk_f2100_m', 'kk_f2100_w', 'kk_f2100_c1_3686', 'kk_f2100_c2_3686', 'kk_f2100_t1_3686', 'kk_f2100_t2_3686', 'kk_f1950_m', 'kk_f2150_m', 'kk_f2340_m', 'kk_f1950_w', 'kk_f2150_w', 'kk_f2340_w', 'kk_f1950_c1_3686', 'kk_f1950_c2_3686', 'kk_f1950_c3_3686', 'kk_f1950_c4_3686', 'kk_f1950_c5_3686', 'kk_f2150_c1_3686', 'kk_f2150_c2_3686', 'kk_f2150_c3_3686', 'kk_f2150_c4_3686', 'kk_f2150_c5_3686', 'kk_f2340_c1_3686', 'kk_f2340_c2_3686', 'kk_f2340_c3_3686', 'kk_f2340_c4_3686', 'kk_f2340_c5_3686', 'kk_f1950_t1_3686', 'kk_f1950_t2_3686', 'kk_f1950_t3_3686', 'kk_f1950_t4_3686', 'kk_f1950_t5_3686', 'kk_f2150_t1_3686', 'kk_f2150_t2_3686', 'kk_f2150_t3_3686', 'kk_f2150_t4_3686', 'kk_f2150_t5_3686', 'kk_f2340_t1_3686', 'kk_f2340_t2_3686', 'kk_f2340_t3_3686', 'kk_f2340_t4_3686', 'kk_f2340_t5_3686', 'kk_K2400_m', 'kk_K2400_w', 'kk_K2400_c1_3686', 'kk_K2400_t1_3686', 'kk_K1830_m', 'kk_K2250_m', 'kk_K1830_w', 'kk_K2250_w', 'kk_K1830_c1_3686', 'kk_K1830_c2_3686', 'kk_K1830_c3_3686', 'kk_K1830_c4_3686', 'kk_K2250_c1_3686', 'kk_K2250_c2_3686', 'kk_K2250_c3_3686', 'kk_K2250_c4_3686', 'kk_K1830_t1_3686', 'kk_K1830_t2_3686', 'kk_K1830_t3_3686', 'kk_K1830_t4_3686', 'kk_K2250_t1_3686', 'kk_K2250_t2_3686', 'kk_K2250_t3_3686', 'kk_K2250_t4_3686', 'inta_c_3773', 'inta_t_3773', 'kk_f980_c1_3773', 'kk_f980_c2_3773', 'kk_f980_t1_3773', 'kk_f980_t2_3773', 'kk_f1270_c1_3773', 'kk_f1270_c2_3773', 'kk_f1270_c3_3773', 'kk_f1270_c4_3773', 'kk_f1270_c5_3773', 'kk_f1270_t1_3773', 'kk_f1270_t2_3773', 'kk_f1270_t3_3773', 'kk_f1270_t4_3773', 'kk_f1270_t5_3773', 'kk_f1525_c1_3773', 'kk_f1525_c2_3773', 'kk_f1525_c3_3773', 'kk_f1525_c4_3773', 'kk_f1525_c5_3773', 'kk_f1525_t1_3773', 'kk_f1525_t2_3773', 'kk_f1525_t3_3773', 'kk_f1525_t4_3773', 'kk_f1525_t5_3773', 'kk_f2470_c1_3773', 'kk_f2470_c2_3773', 'kk_f2330_c1_3773', 'kk_f2330_c2_3773', 'kk_f1710_c1_3773', 'kk_f1710_c2_3773', 'kk_f2470_t1_3773', 'kk_f2470_t2_3773', 'kk_f2330_t1_3773', 'kk_f2330_t2_3773', 'kk_f1710_t1_3773', 'kk_f1710_t2_3773', 'kk_f0long_c1_3773', 'kk_f0long_c2_3773', 'kk_f0long_t1_3773', 'kk_f0long_t2_3773', 'kk_f2100_c1_3773', 'kk_f2100_c2_3773', 'kk_f2100_t1_3773', 'kk_f2100_t2_3773', 'kk_f1950_c1_3773', 'kk_f1950_c2_3773', 'kk_f1950_c3_3773', 'kk_f1950_c4_3773', 'kk_f1950_c5_3773', 'kk_f2150_c1_3773', 'kk_f2150_c2_3773', 'kk_f2150_c3_3773', 'kk_f2150_c4_3773', 'kk_f2150_c5_3773', 'kk_f2340_c1_3773', 'kk_f2340_c2_3773', 'kk_f2340_c3_3773', 'kk_f2340_c4_3773', 'kk_f2340_c5_3773', 'kk_f1950_t1_3773', 'kk_f1950_t2_3773', 'kk_f1950_t3_3773', 'kk_f1950_t4_3773', 'kk_f1950_t5_3773', 'kk_f2150_t1_3773', 'kk_f2150_t2_3773', 'kk_f2150_t3_3773', 'kk_f2150_t4_3773', 'kk_f2150_t5_3773', 'kk_f2340_t1_3773', 'kk_f2340_t2_3773', 'kk_f2340_t3_3773', 'kk_f2340_t4_3773', 'kk_f2340_t5_3773', 'kk_K2400_c1_3773', 'kk_K2400_t1_3773', 'kk_K1830_c1_3773', 'kk_K1830_c2_3773', 'kk_K1830_c3_3773', 'kk_K1830_c4_3773', 'kk_K2250_c1_3773', 'kk_K2250_c2_3773', 'kk_K2250_c3_3773', 'kk_K2250_c4_3773', 'kk_K1830_t1_3773', 'kk_K1830_t2_3773', 'kk_K1830_t3_3773', 'kk_K1830_t4_3773', 'kk_K2250_t1_3773', 'kk_K2250_t2_3773', 'kk_K2250_t3_3773', 'kk_K2250_t4_3773']
        mod_name_list = ['phif0_980_3686', 'phif2_1270_3686', 'phif2_1525_3686', 'phif0_2470_3686', 'phif0_2330_3686', 'phif0_1710_3686', 'phif0_long_3686', 'phif0_2100_3686', 'phif2_1950_3686', 'phif2_2150_3686', 'phif2_2340_3686', 'K2400_3686', 'K2400_3686', 'K1830_3686', 'K2250_3686', 'K1830_3686', 'K2250_3686', 'phif0_980_3773', 'phif2_1270_3773', 'phif2_1525_3773', 'phif0_2470_3773', 'phif0_2330_3773', 'phif0_1710_3773', 'phif0_long_3773', 'phif0_2100_3773', 'phif2_1950_3773', 'phif2_2150_3773', 'phif2_2340_3773', 'K2400_3773', 'K2400_3773', 'K1830_3773', 'K2250_3773', 'K1830_3773', 'K2250_3773']
        for json_addr in json_addr_list:
            filename = glob.glob(os.path.join("config/",json_addr+"*"))
            with open(filename[0], encoding='utf-8') as f:
                pwa_json = json.loads(f.read())
                mod_info = pwa_json["mod_info"]
                # mod_index = [n for n, mod in enumerate(mod_info) if re.match(".*"+"_".join(mod["mod"].split("_")[0:-1])+".*"," ".join(mod_name_list))]
                # 
                # mod_index = [n for n, mod in enumerate(mod_info) if re.match(".*" + mod["mod"].replace("_kk", "") + ".*", " ".join(mod_name_list))]
                # 
                mod_index = [n for n, mod in enumerate(mod_info) if mod["mod"].replace("_kk", "") in mod_name_list]
                mod_info = [mod_info[i] for i in mod_index]
                for i in range(len(name_list)):
                    for mod in mod_info:
                        for arg_key in mod["args"].keys():
                            if name_list[i] == arg_key:
                                mod["args"][arg_key]["value"] = float(np_values[i])
                                mod["args"][arg_key]["error"] = float(np_errors[i])
                _pwa_json = {"mod_info":mod_info} 
                json_str = json.dumps(_pwa_json, indent=4)
                with open(save_addr + "/{0}.{1}".format(json_addr, n), 'w') as json_file:
                    json_file.write(json_str)
        with open(save_addr + "/result_info.json", 'w') as f:
            json.dump(result_info,f)
