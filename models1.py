# Copyright 2018 The GraphNets Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
"""Model architectures for the demos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import modules
from graph_nets import utils_tf
import sonnet as snt
import tensorflow as tf
import numpy as np
import graph_nets

NUM_LAYERS = 1 # Hard-code number of layers in the edge/node/global models.
LATENT_SIZE = 16  # Hard-code latent layer sizes for demos.

def make_mlp_model():
    return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE] * NUM_LAYERS, activate_final=True),
      snt.LayerNorm()
  ])

def _make_cnn_model(): 
    return snt.Sequential([
      snt.Conv1D(output_channels=16, kernel_shape=3, stride=1, rate=1, padding='SAME', use_bias=True, data_format='NWC'),
      snt.Conv1D(output_channels=16, kernel_shape=3, stride=1, rate=1, padding='SAME', use_bias=True, data_format='NWC'),
      snt.LayerNorm()        
     ])
 #return snt.Conv1D(10,3)


class MLPGraphIndependent(snt.AbstractModule):
    """GraphIndependent with MLP edge, node, and global models."""

    def __init__(self, name="MLPGraphIndependent"):
        super(MLPGraphIndependent, self).__init__(name=name)
        with self._enter_variable_scope():
            self._network = modules.GraphIndependent(
                edge_model_fn=_make_cnn_model,
                node_model_fn=_make_cnn_model,
                global_model_fn=_make_cnn_model
          )
            self._network1 = modules.GraphIndependent(
                edge_model_fn=make_mlp_model,
                node_model_fn=make_mlp_model,
                global_model_fn=make_mlp_model
          )

    def _build(self, inputs):
        temp1=[]
        for idx,v in enumerate(inputs):
            if(idx==0):
                new_v=tf.expand_dims(v,1)
                temp1.append(new_v)
            elif(idx==1):
                new_v=tf.expand_dims(v,1)
                temp1.append(new_v)
            elif(idx==2):
                temp1.append(v)
            elif(idx==3):
                temp1.append(v)
            elif(idx==4):
                new_v=tf.expand_dims(v,1)
                temp1.append(new_v)
            elif(idx==5):
                temp1.append(v)
            else:
                temp1.append(v)
        inputs=graph_nets.graphs.GraphsTuple(*temp1)
        temp=self._network(inputs)
        temp1=[]
        for idx,v in enumerate(temp):
            if(idx==0):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==1):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==2):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            elif(idx==3):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            elif(idx==4):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==5):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            else:
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
        temp=graph_nets.graphs.GraphsTuple(*temp1)
        return self._network1(temp)

class MLPGraphNetwork(snt.AbstractModule):
  """GraphNetwork with MLP edge, node, and global models."""

  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    with self._enter_variable_scope():
        self._network = modules.GraphNetwork(_make_cnn_model, _make_cnn_model, _make_cnn_model)
  def _build(self, inputs):
    temp1=[]
    for idx,v in enumerate(inputs):
        if(idx==0):
            new_v=tf.expand_dims(v,1)
            temp1.append(new_v)
        elif(idx==1):
            new_v=tf.expand_dims(v,1)
            temp1.append(new_v)
        elif(idx==2):
            temp1.append(v)
        elif(idx==3):
            temp1.append(v)
        elif(idx==4):
            new_v=tf.expand_dims(v,1)
            temp1.append(new_v)
        elif(idx==5):
            temp1.append(v)
        else:
            temp1.append(v)
    inputs=graph_nets.graphs.GraphsTuple(*temp1)
    return self._network(inputs)

class EncodeProcessDecode(snt.AbstractModule):
  """Full encode-process-decode model.

  The model we explore includes three components:
  - An "Encoder" graph net, which independently encodes the edge, node, and
    global attributes (does not compute relations etc.).
  - A "Core" graph net, which performs N rounds of processing (message-passing)
    steps. The input to the Core is the concatenation of the Encoder's output
    and the previous output of the Core (labeled "Hidden(t)" below, where "t" is
    the processing step).
  - A "Decoder" graph net, which independently decodes the edge, node, and
    global attributes (does not compute relations etc.), on each message-passing
    step.

                      Hidden(t)   Hidden(t+1)
                         |            ^
            *---------*  |  *------*  |  *---------*
            |         |  |  |      |  |  |         |
  Input --->| Encoder |  *->| Core |--*->| Decoder |---> Output(t)
            |         |---->|      |     |         |
            *---------*     *------*     *---------*
  """

  def __init__(self,
               edge_output_size=None,
               node_output_size=None,
               global_output_size=None,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    # Transforms the outputs into the appropriate shapes.
    if edge_output_size is None:
      edge_fn = None
    else:
      edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
    if node_output_size is None:
      node_fn = None
    else:
      node_fn = lambda: snt.Linear(node_output_size, name="node_output")
    if global_output_size is None:
      global_fn = None
    else:
      global_fn = lambda: snt.Linear(global_output_size, name="global_output")
    with self._enter_variable_scope():
      self._output_transform = modules.GraphIndependent(edge_fn, node_fn,
                                                        global_fn)

  def _build(self, input_op, num_processing_steps): 
    latent1 = self._encoder(input_op)
    print(latent1)
    output_ops = []    
    latent0 = latent1
    for i in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent1], axis=1)
        latent1 = self._core(core_input)
        temp1=[]
        for idx,v in enumerate(latent1):
            if(idx==0):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==1):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==2):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            elif(idx==3):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            elif(idx==4):
                new_v=tf.reshape(v,[-1,16])
                temp1.append(new_v)
            elif(idx==5):
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
            else:
                new_v=tf.reshape(v,[-1])
                temp1.append(new_v)
        latent1=graph_nets.graphs.GraphsTuple(*temp1)
        decoded_op = self._decoder(latent1)
        output_ops.append(self._output_transform(decoded_op))
    return output_ops
