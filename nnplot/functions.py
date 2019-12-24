"""
- Copyright (c) 2019 RAD Ltd.
- Contact: Yuval.A (Rad R&D) yuval_a@rad.com
-
- The MIT License
-
- Copyright (c) 2019 RAD Ltd.
-
- Permission is hereby granted, free of charge, to any person obtaining a copy
- of this software and associated documentation files (the "Software"), to deal
- in the Software without restriction, including without limitation the rights
- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
- copies of the Software, and to permit persons to whom the Software is
- furnished to do so, subject to the following conditions:
-
- The above copyright notice and this permission notice shall be included in
- all copies or substantial portions of the Software.
-
- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
- THE SOFTWARE.



Parts of this code (mainly at the "plot_net" function) are copied and modified
from ann_visualizer project https://github.com/Prodicode/ann-visualizer.

Related Copyrights:

<ann_visualizer related copyrights>
    - The MIT License
    -
    - Copyright (c) 2018 Tudor Gheorghiu;
    -
    - Permission is hereby granted, free of charge, to any person obtaining a copy
    - of this software and associated documentation files (the "Software"), to deal
    - in the Software without restriction, including without limitation the rights
    - to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    - copies of the Software, and to permit persons to whom the Software is
    - furnished to do so, subject to the following conditions:
    -
    - The above copyright notice and this permission notice shall be included in
    - all copies or substantial portions of the Software.
    -
    - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    - IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    - FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    - AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    - LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    - OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
    - THE SOFTWARE.
</ann_visualizer related>



Version 1.0
This project is a work in progress.

Known Issues/Limitations/TODOs:
    prune:
    - Node choice process at pruning stage is not optimal (see choose_top function)
    - Currently only supports pruning for visualization, and no actual
      additional use, because default attributes are set for the new model's
      layers, and not the original model's ones.
    - Currently only supports the same max limit for all layers.

    plot_net:
    - Seems like the edge coloring loop returns to the start for each loop.
    - Currently only supports the same max limit for all layers.
"""


from graphviz import Digraph
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np

def calc_weights(model):
    """
    Creates a clean (no weights) NN of the desired dimensions.
    # Arguments:
        model: A Keras model instance.
        max_limit: maximal number of nodes on each layer.

    Legend:
    * L_w: matrix of weights for all nodes on all layers
    * W_i: (input) edges weights for layer i
    * l_i: calculated weights for nodes on layer i
    * n_w: dummy output layer that represents the summed up influence of each node on its previous layer on output nodes
    """

    # Find the size of the biggest layer.
    # All the other layers will be pad with zeros to match its size.
    max_layer_size = 0
    for layer in model.layers:
        # Compared both input and output sizes to take input and output layers into consideration.
        max_layer_size = max(max_layer_size, layer.input_shape[1],layer.output_shape[1])

    # Initial input layer weights (fixed later)
    L_w = np.zeros([max_layer_size, 1]) # Start with max_layer_size in order to pad with zeros.
    L_w[0:model.layers[0].input_shape[1]] = 1 # replace needed nodes to 1. The rest will be left as zeros (padding)

    ## Note about padding with zeros:
    # Padding with zeros is OK in this case because
    # multiplying by zero will cancel the effect of the edge
    # between the place holder node and the receiving, next layer's, node.
    # Also, it will not be taken under account when choosing the top k performing notes
    # because all weights are positive.

    # Calculate weights for all layers (except the output layer)
    for i in range(0,len(model.layers)-1):
        W_i = np.matrix(model.layers[i].get_weights()[0])
        W_i = np.absolute(W_i)                               # Take as positive weights in order to compare influence
        real_i_layer_size = model.layers[i].input_shape[1]
        l_i = np.dot(L_w[0:real_i_layer_size,-1].T, W_i)     # Calc L_i layer's weights
        if l_i.shape[0] == 1:                                # Make sure all layers are added as columns to L_w
            l_i = l_i.T
        diff = (max_layer_size - l_i.shape[0])
        if diff > 0:
            l_i = np.vstack((l_i,np.zeros([diff,1])))        # Stack into L_w
        L_w = np.hstack((L_w,l_i))                           # Stack result into L_w

    # output layer:
    W_i = np.matrix(model.layers[-1].get_weights()[0])       # Weights of last layer
    W_i = np.absolute(W_i)                                   # Take as positive weights in order to compare influence
    real_i_layer_size = model.layers[-1].input_shape[1]
    l_i = np.multiply(L_w[0:real_i_layer_size, -1],W_i)      # Weight of each node w.r.t. the output layer
    n_w = np.add(l_i[:,0],l_i[:,1])                          # Sum the output columns
    diff = (max_layer_size - n_w.shape[0])
    if diff > 0:
        n_w = np.vstack((n_w,np.zeros([diff,1])))            # Stack into L_w
    L_w = np.hstack((L_w,n_w))

    # Fix input layer's weights and size
    if L_w[0].shape[1] > 1:
        W_0 = np.matrix(model.layers[0].get_weights()[0])
        W_0 = np.absolute(W_0)                               # Taken as positive weights in order to compare influence
        real_i_layer_size = model.layers[1].input_shape[1]
        L_w[:,0] = np.dot(W_0, L_w[0:real_i_layer_size,1])

    return L_w


def choose_top(L_w, max_limit):
    """
    Pick top k preforming nodes

    version 1.0 - currently only chooses the top weighted nodes.
    on each layer (after weight distribution, see calc_weights function).
    TODO - choose more informative nodes from each layer.

    # Arguments
      L_w: matrix of weights for all nodes on all layers.
      max_limit: maximal number of nodes on each layer.
    """
    return np.argsort(L_w, axis=0)[-max_limit:,:]           # Also works when max_limit > layer size

def create_clean_net(model, max_limit):
    """
    Creates a clean (no weights) NN of the desired dimensions.
    # Arguments
        model: A Keras model instance.
        max_limit: maximal number of nodes on each layer.
    """
    network = Sequential()

    # Add input & first Hidden Layer
    network.add(Dense(units=min(max_limit,model.layers[0].output_shape[1]),
                      activation='relu',
                      kernel_initializer='uniform',
                      input_dim=min(max_limit,model.layers[0].input_shape[1])))

    # Add the rest of the Hidden Layers
    for t in range(1,len(model.layers) - 1):
        # Hidden Layer #t
        network.add(Dense(units=min(max_limit,model.layers[t].output_shape[1]),
                          activation='relu',
                          kernel_initializer='uniform'))

    # Add the output Layer
    network.add(Dense(units=min(max_limit,model.layers[-1].output_shape[1]),
                      activation='sigmoid',
                      kernel_initializer='uniform'))


    # Compile model - (just for plotting the model)
    myOptimizer = sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    return network


def prune(model, max_limit=-1, verbose=True, input_list=None):

    """
    Resize a Sequential model.

    Version 1.0 limitations:
    1. Currently only supports pruning for visualization, and no actual
      additional use, because default attributes are set for the new model's
      layers, and not the original model's ones.
      TODO: copy original original activations to the pruned model.
        <hint?>
        https://keras.io/layers/about-keras-layers/
        layer.get_config(): returns a dictionary containing the configuration of the layer.
        The layer can be re-instantiated from its config via:
        layer = Dense(32)
        config = layer.get_config()
        reconstructed_layer = Dense.from_config(config)
        </hint?>
    2. Currently only supports the same max limit for all layers.

    # Arguments
        model: A Keras model instance.
        max_limit: maximal number of nodes on each layer.
        verbose: print inforamtion along the way.
        input_list: list of input names, so when the input_indexes output will have their names.
    # Outputs
        network: the new pruned network.
        input_indexes: the indexes of the chosen inputs
    """

    # If there is a max limit resize each model's layer to keep top k preforming nodes
    if max_limit > 0:
        if verbose:
            print "\nVerbose mode is ON (by default)\n - to turn off, call prune with: verbose=False"

        L_w = calc_weights(model)
        if verbose:
            for i in range(L_w[0,:].shape[1]):
                print "layer %s : " % i
                print L_w[:,i]
                print "--"

        # Take k top values of each column (layer)
        top_picks = choose_top(L_w, max_limit)
        if verbose:
            print "\nTOP PICKS:\n * each column is a layer in the NN\n * each cell is the chosen node's index\n\n" + str(top_picks)

        # Create a new model
        network = create_clean_net(model, max_limit)
        if verbose:
            # print model's old/new dimensions
            print "\n\nModel Dimensions Change\n\nBefore: "
            model.summary()
            print "\nAfter: "
            network.summary()

        # Fill in the newly created layers with the weights of the selected nodes
        # a layer's weights is of the form (A,B)
        # - A is a matrix
        # - B is a vector

        for l in range(len(model.layers) - 1):             #   For each layer
            weights = model.layers[l].get_weights()        #   Get weights

            A = np.empty([1, top_picks[:,l].shape[0]])     #   Create empty matrix (start with only first raw)
            for i in range(top_picks.shape[0]):            #   TODO for min() ?
                node_index = top_picks[i,l]                #   Get the node's index
                A_i = weights[0][node_index]               #   Get all it's weights
                picks_of_Ai = A_i[top_picks[:,l+1]].T[0]   #   Select the weights of the edges from the node to all nodes on the next level
                A = np.vstack((A, picks_of_Ai))            #   Add it (as a raw) to the A matrix
            B = weights[1][top_picks[:,l] - 1].T           #   Fill B with the output weights
            weights_new = [A[1:,:], B[0]]                  #   Stock them together drop the first A layer (garbage)
            network.layers[l].set_weights(weights_new)     #   Fill layer

        if input_list:
            input_indexes = top_picks[:,0].T.tolist()[0]         # (list of) indexes of chosen inputs
            input_list = [input_list[i] for i in input_indexes]  #
            if verbose:
                print "\npicked inputs: %s" % input_list
        # else returns [network, None]

    return [network, input_list]
# End of prune()



def plot_net(model, input_list="No inputs titles specified",  view=True, filename="network.gv", plot_title="My Neural Network", out_title ="No output title specified", color_edges="rb", print_weights=False, size_limit=10):
    """ Visualizes a Sequential model.
    # Arguments
        model: A Keras model instance.
        view: whether to display the model after generation.
        filename: where to save the visualization. (a .gv file).
        title: A title for the graph.
        color_edges: whether to visualize the weights of the edges.
                     options:
                        "rb" - red / black: red for positive edges and black for negative ones.
                        "mc" - multi color: all edges that converge into the same node, have the same (unique) color.
                        "none" - all edges painted black.
        print_weights: whether to print the weights of the edges to the screen.
        size_limit: max number of nodes in each layer.
    """

    input_layer       = 0
    hidden_layers_nr  = 0
    layer_types       = []
    hidden_layers     = []
    output_layer      = 0
    total_prev_nodes  = 0

    # Get nodes weights to illustrate as sizes
    node_weights = calc_weights(model)

    plot_title += "\n\nInputs List:\n\n" + "\n".join(input_list)

    if color_edges not in ["rb","mc","none"]:
        print  "WARNING: color_edges value does not match possible options.\
               \n'none' was chosen for you. Please retry while using these options:\
               \n * 'rb' - red / black: red for positive edges and black for negative ones\
               \n * 'mc' - multi color: all edges that converge into the same node, have the same (unique) color\
               \n * 'none' - all edges painted black"

    for layer in model.layers:

        if(layer == model.layers[0]):
            input_layer = int(str(layer.input_shape).split(",")[1][1:-1])
            hidden_layers_nr += 1
            if (type(layer) == keras.layers.core.Dense):
                hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]))
                layer_types.append("Dense")
            else:
                hidden_layers.append(1)
                if (type(layer) == keras.layers.convolutional.Conv2D):
                    layer_types.append("Conv2D")
                elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                    layer_types.append("MaxPooling2D")
                elif (type(layer) == keras.layers.core.Dropout):
                    layer_types.append("Dropout")
                elif (type(layer) == keras.layers.core.Flatten):
                    layer_types.append("Flatten")
                elif (type(layer) == keras.layers.core.Activation):
                    layer_types.append("Activation")
        else:
            if(layer == model.layers[-1]):
                output_layer = int(str(layer.output_shape).split(",")[1][1:-1])
            else:
                hidden_layers_nr += 1
                if (type(layer) == keras.layers.core.Dense):
                    hidden_layers.append(int(str(layer.output_shape).split(",")[1][1:-1]))
                    layer_types.append("Dense")
                else:
                    hidden_layers.append(1)
                    if (type(layer) == keras.layers.convolutional.Conv2D):
                        layer_types.append("Conv2D")
                    elif (type(layer) == keras.layers.pooling.MaxPooling2D):
                        layer_types.append("MaxPooling2D")
                    elif (type(layer) == keras.layers.core.Dropout):
                        layer_types.append("Dropout")
                    elif (type(layer) == keras.layers.core.Flatten):
                        layer_types.append("Flatten")
                    elif (type(layer) == keras.layers.core.Activation):
                        layer_types.append("Activation")
        last_layer_nodes = input_layer
        nodes_up = input_layer
        if(type(model.layers[0]) != keras.layers.core.Dense):
            last_layer_nodes = 1
            nodes_up         = 1
            input_layer      = 1

        g = Digraph('g', filename=filename)
        n = 0
        g.graph_attr.update(splines="false", nodesep='1', ranksep='2')

        #Input Layer
        with g.subgraph(name='cluster_input') as c:
            if(type(model.layers[0]) == keras.layers.core.Dense):
                the_label = plot_title+'\n\n\n\nInput Layer'
                if color_edges == "rb": # add color legend
                    the_label += "\nBlack color - negative weight\nRed color - positive weight"
                if (int(str(model.layers[0].input_shape).split(",")[1][1:-1]) > size_limit):
                    the_label += " (+"+str(int(str(model.layers[0].input_shape).split(",")[1][1:-1]) - size_limit)+")"
                    input_layer = size_limit
                c.attr(color='white')
                for i in range(0, input_layer):
                    n += 1
                    c.node(str(n))
                    c.attr(label=the_label)
                    c.attr(rank='same')
                    c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#707070", shape="circle")

            elif(type(model.layers[0]) == keras.layers.convolutional.Conv2D):
                #Conv2D Input visualizing
                the_label = plot_title+'\n\n\n\nInput Layer'
                c.attr(color="white", label=the_label)
                c.node_attr.update(shape="square")
                pxls = str(model.layers[0].input_shape).split(',')
                clr = int(pxls[3][1:-1])
                if (clr == 1):
                    clrmap = "Grayscale"
                    the_color = "black:white"
                elif (clr == 3):
                    clrmap = "RGB"
                    the_color = "#e74c3c:#3498db"
                else:
                    clrmap = ""
                c.node_attr.update(fontcolor="white", fillcolor=the_color, style="filled")
                n += 1
                c.node(str(n), label="Image\n"+pxls[1]+" x"+pxls[2]+" pixels\n"+clrmap, fontcolor="white")
            else:
                raise ValueError("Visualizer: Layer not supported for visualizing")

        # Hidden Layers
        for i in range(0, hidden_layers_nr):

            with g.subgraph(name="cluster_"+str(i+1)) as c:
                if (layer_types[i] == "Dense"):
                    c.attr(color='white')
                    c.attr(rank='same')
                    #If hidden_layers[i] > size_limit, dont include all
                    the_label = ""
                    if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > size_limit):
                        the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - size_limit)+")"
                        hidden_layers[i] = size_limit
                    c.attr(labeljust="right", labelloc="b", label=the_label)

                    for j in range(0, hidden_layers[i]):
                        n += 1
                        c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#707070")

######################################################################################################
###########################                 Color The Edges                ###########################
######################################################################################################

                        if color_edges == "mc":
                            node_color_plate = np.random.randint(255, size=(3))
                        orig_weights  = model.layers[i].get_weights()[0]
                        layer_weights = np.absolute(orig_weights)                                                     # Get the layers weights
                        layer_weights = np.concatenate((layer_weights, [np.zeros(layer_weights.shape[1])]))           # Add a zeros to the array
                        layer_weights = np.interp(layer_weights, (layer_weights.min(), layer_weights.max()), (0, 10)) # Normalize to (0-10)
                        layer_weights = layer_weights[:-1]                                                            # Remove the zero
                        for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                            edge_col = '#081d30' # Defult color is black. If coloring method won't be specified, it will remain black.
                            if color_edges == "mc":
                                red    = int(node_color_plate[0])
                                green  = int(node_color_plate[1])
                                blue   = int(node_color_plate[2])
                                rgb = (red,green,blue)
                                hex_result = "".join([format(val, '02X') for val in rgb])
                                edge_col = '#' + hex_result

                            elif color_edges == "rb":
                                if (orig_weights[h - (nodes_up - last_layer_nodes + 1)][j]) > 0 :
                                    edge_col = '#e24e28' # red

                            edge_weight = layer_weights[h - (nodes_up - last_layer_nodes + 1)][j]

                            g.edge(str(h), str(n), color=edge_col, fontcolor="#707070",
                            label="",fontsize='2', penwidth=str(edge_weight))

                            if print_weights:
                                print "edge [%i]-->[%i]: weight = %s" % (h,n,str(edge_weight))

######################################################################################################
######################################################################################################

                    last_layer_nodes = hidden_layers[i]
                    nodes_up += hidden_layers[i]
                elif (layer_types[i] == "Conv2D"):
                    c.attr(style='filled', color='#5faad0')
                    n += 1
                    kernel_size = str(model.layers[i].get_config()['kernel_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['kernel_size']).split(',')[1][1 : -1]
                    filters = str(model.layers[i].get_config()['filters'])
                    c.node("conv_"+str(n), label="Convolutional Layer\nKernel Size: "+kernel_size+"\nFilters: "+filters, shape="square")
                    c.node(str(n), label=filters+"\nFeature Maps", shape="square")
                    g.edge("conv_"+str(n), str(n))
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), "conv_"+str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "MaxPooling2D"):
                    c.attr(color="white")
                    n += 1
                    pool_size = str(model.layers[i].get_config()['pool_size']).split(',')[0][1] + "x" + str(model.layers[i].get_config()['pool_size']).split(',')[1][1 : -1]
                    c.node(str(n), label="Max Pooling\nPool Size: "+pool_size, style="filled", fillcolor="#8e44ad", fontcolor="white")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Flatten"):
                    n += 1
                    c.attr(color="white")
                    c.node(str(n), label="Flattening", shape="invtriangle", style="filled", fillcolor="#2c3e50", fontcolor="white")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Dropout"):
                    n += 1
                    c.attr(color="white")
                    c.node(str(n), label="Dropout Layer", style="filled", fontcolor="white", fillcolor="#f39c12")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1
                elif (layer_types[i] == "Activation"):
                    n += 1
                    c.attr(color="white")
                    fnc = model.layers[i].get_config()['activation']
                    c.node(str(n), shape="octagon", label="Activation Layer\nFunction: "+fnc, style="filled", fontcolor="white", fillcolor="#00b894")
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        g.edge(str(h), str(n))
                    last_layer_nodes = 1
                    nodes_up += 1

        with g.subgraph(name='cluster_output') as c:
            if (type(model.layers[-1]) == keras.layers.core.Dense):
                c.attr(color='white')
                c.attr(rank='same')
                c.attr(labeljust="1")
                for i in range(1, output_layer+1):
                    n += 1
                    c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#707070")

######################################################################################################
###########################                 Color The Edges                ###########################
######################################################################################################

                    if color_edges == "mc":
                        node_color_plate = np.random.randint(255, size=(3))
                    orig_weights  = model.layers[i].get_weights()[0]
                    layer_weights = np.absolute(orig_weights)                                                     # Get the layers weights
                    layer_weights = np.concatenate((layer_weights, [np.zeros(layer_weights.shape[1])]))           # Add a zeros to the array
                    layer_weights = np.interp(layer_weights, (layer_weights.min(), layer_weights.max()), (0, 10)) # Normalize to (0-10)
                    layer_weights = layer_weights[:-1]                                                            # Remove the zero
                    for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                        edge_col = '#081d30' # Default color is black. If coloring method won't be specified, it will remain black.
                        if color_edges == "mc":
                            red    = int(node_color_plate[0])
                            green  = int(node_color_plate[1])
                            blue   = int(node_color_plate[2])
                            rgb = (red,green,blue)
                            hex_result = "".join([format(val, '02X') for val in rgb])
                            edge_col = '#' + hex_result

                        elif color_edges == "rb":
                            if (orig_weights[h - (nodes_up - last_layer_nodes + 1)][j]) > 0 :
                                edge_col = '#e24e28' # red

                        edge_weight = layer_weights[h - (nodes_up - last_layer_nodes + 1)][j]

                        g.edge(str(h), str(n), color=edge_col, fontcolor="#707070",
                        label="",fontsize='2', penwidth=str(edge_weight))

                        if print_weights:
                            print "edge [%i]-->[%i]: weight = %s" % (h,n,str(edge_weight))

######################################################################################################
######################################################################################################

                c.attr(label='Output Layer' + out_title, labelloc="bottom")
                c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#707070", shape="circle")

        g.attr(arrowShape="none")
        g.edge_attr.update(arrowhead="none")
        if view == True:
            g.view()
