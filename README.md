# nnplot
### A Python library for pruning and visualizing Keras Neural Networks' structure and weights.
![cover image](https://github.com/Yuval-Ai/nnplot/blob/master/Images/banner.png)

**nnplot** is a Python library for visualizing Neural Networks in an informative representation. 

It has the ability to display the NN's structure, dominance of links weights per layer and their polarity (positive/negative).

It also provides functions for pruning the NN in order to display the **n** “most important” nodes of each layer.

* [How to install it?](##How-to-install-it?) 

* [How to use it?](##How-to-use-it?)

* [Start with an example](##Start-with-an-examlpe)

* [Additional Information](##Additional-Information)



## How to install it?

nnplot is available via pip:

```bash
  pip install nnplot
```
/TODO - dependencies?

*alternatively, you can download the *nnplot*  subfolder and place it in the same directory of your code. 


## How to use it?
once you have a built and trained Keras model, you can simply:

**plot** the model:

```python
from nnplot.functions import plot_net

plot_net(model)
```

or **prune** the model:

```python
from nnplot.functions import prune

[new_model, new_input_list] = prune(model, max_limit=5)
```



To make the most out of the functions mentioned above, try using their optional flags (use examples follows):

**plot:**

```python
from nnplot.functions import plot_net

plot_net(model, 
         input_list=['1: Input A', '2: Input B', '3: Input C','etc..'],
         view=True,
         filename="my_network.gv",
         plot_title="My Neural Network Title",
         out_title ="This is my output title: \n 1. output A \n 2. output B",
         color_edges="rb",
         print_weights=False,
         size_limit=10)
```

Arguments:

`model`: A Keras model instance.
`view`: whether to plot the model on screen after its generation.
`filename`: path and name to save the visualization outcome, as a *PDF* and a *.gv* (graph-viz) file.
`title`: A title for the graph.
`color_edges`: whether to visualize the weights of the edges as colors.
	options:

	- "*rb*" - Red / Black: red for positive edges and black for negative ones.
	- "*mc*" - Multi Colored: all edges that converge into the same node, have the same (unique) color.
	- "*none*" - all edges painted black (but thickness visualization remains).
`print_weights`: whether to print the weights of the edges to the screen.
`size_limit`: max number of nodes in each layer (simply the first *n* nodes, use **prune** for a more complex node selection).

**prune:**

```python
from nnplot.functions import prune

[new_network, new_input_list] = prune(model,
                                      max_limit,
                                      input_list=['1: Input A', '2: Input B','etc..'],
                                      verbose=True)
				
```

Arguments:
`model`: A Keras model instance.
`max_limit`: maximal number of nodes on each layer. ([How are the nodes picked in this prune?](\TODO))
`input_list`: list of input names, so that the new *input_indexes* output will have their original names.
`verbose`: print information of the process along the way.

Outputs:
`network`: the new pruned network.
`input_indexs`: the indexes of the chosen inputs



#### Examples:

##### 1. plot <\TODO - with something>:

```python
\TODO code...
```

**result:** 

\TODO - Image 

##### 2. plot <\TODO - with something>:

```python
\TODO code...
```

**result:** 

\TODO - Image 

##### 3. plot <\TODO - with something>:

```python
\TODO code...
```

**result:** 

\TODO - Image 

##### 4. plot <\TODO - with something>:

```python
\TODO code...
```

**result:** 

\TODO - Image 



## Start with an example

## Additional Information
