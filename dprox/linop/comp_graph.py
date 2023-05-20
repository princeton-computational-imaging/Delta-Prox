import copy as cp
from collections import defaultdict

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigs

from .constant import Constant
from .edge import Edge
from .sum import copy
from .variable import Variable
from .vstack import split, vstack
from .base import LinOp


class CompGraph:
    """A computation graph representing a composite lin op.
    """

    instanceCnt = 0

    def __init__(self, end, zero_out_constant=False):
        self.instanceID = CompGraph.instanceCnt
        CompGraph.instanceCnt += 1

        # TODO: can we deepcopy directly, then we can remove the cp.copy below as well.
        # self.end = cp.copy(end).to(end.device)
        self.end = end

        # Construct via graph traversal.
        self.nodes = []
        self.edges = []
        self.constants = []
        self.input_edges = {}
        self.output_edges = {}
        new_vars = []

        # Assumes all nodes have at most one output.
        ready = [self.end]
        done = []
        node_to_copies = {}
        self.split_nodes = {}

        while len(ready) > 0:
            curr = ready.pop(0)
            done.append(curr)
            if isinstance(curr, Variable):
                # new_vars may contain specific variables more than once
                new_vars.append(curr)
            # Zero out constants.
            self.nodes.append(curr)

            input_edges = []
            for node in curr.input_nodes:
                # Zero out constants. Constants are handled in absorb_offset
                if isinstance(node, Constant):
                    if zero_out_constant:
                        node = Constant(node.value * 0).to(node.device)
                    self.constants.append(node)
                else:
                    # avoid copying too many nodes
                    # if one node is visited more than once, then we need flag it
                    if node not in node_to_copies:
                        # TODO: do we really need copy
                        # cnode = cp.copy(node).to(node.device)
                        cnode = node
                        node_to_copies[node] = cnode
                    else:
                        self.split_nodes[node_to_copies[node]] = True
                    node = node_to_copies[node]

                # collect input edges for current node
                edge = Edge(node, curr)
                input_edges.append(edge)

                # the same edge is the output edges of these start nodes
                if node not in self.output_edges:  # init if not exists
                    self.output_edges[node] = [edge]
                else:
                    self.output_edges[node].append(edge)

                # add child nodes to queue
                if node not in ready and node not in done:
                    ready.append(node)

            self.edges += input_edges
            self.input_edges[curr] = input_edges

        # replace the split nodes with copy nodes
        for n in self.split_nodes.keys():
            # find out the outputs by traverse the outputs of original node
            outedges = self.output_edges[n]
            outnodes = [e.end for e in outedges]
            copy_node = copy(n)

            # link split(copy) node with its original node, link in two direction, forward and adjoint
            self.output_edges[n] = [Edge(n, copy_node)]
            self.input_edges[copy_node] = self.output_edges[n]

            # ---- link the output of copy node to its outputs. ---- #
            self.output_edges[copy_node] = []
            # modify the input edge of the outputs node
            self.nodes.append(copy_node)
            for ns in outnodes:
                inedges = self.input_edges[ns]
                newinedges = []
                for e in inedges:
                    # if input edges's start is original node, replace it with the new copy node.
                    if e.start is n:
                        e = Edge(copy_node, e.end)
                        newinedges.append(e)
                        self.output_edges[copy_node].append(e)
                    else:
                        newinedges.append(e)
                self.input_edges[ns] = newinedges

            # we actually can have more than two input nodes
            copy_node.input_nodes += [n] * (len(self.output_edges[copy_node]) - 1)

        # Make copy node for each variable.
        old_vars = self.end.variables
        id2copy = {}
        copy_nodes = []
        self.var_info = {}
        offset = 0
        for var in old_vars:
            copy_node = copy(var)
            copy_node.orig_node = None
            id2copy[var.uuid] = copy_node
            copy_nodes.append(copy_node)
            # self.var_info[var.uuid] = offset
            # offset += copy_node.size
            self.output_edges[copy_node] = []
            self.nodes.append(copy_node)

        # Replace variables with copy nodes in graph.
        for var in new_vars:
            copy_node = id2copy[var.uuid]
            for output_edge in self.output_edges.get(var, []):
                output_node = output_edge.end
                edge = Edge(copy_node, output_node)
                self.edges.append(edge)
                self.output_edges[copy_node].append(edge)
                idx = self.input_edges[output_node].index(output_edge)
                self.input_edges[output_node][idx] = edge

        # Record information about variables.
        # self.input_size = sum([var.size for var in old_vars])
        # self.output_size = self.end.size

        self.start = split(copy_nodes)
        self.nodes.append(self.start)
        split_outputs = []
        for copy_node in copy_nodes:
            edge = Edge(self.start, copy_node)
            split_outputs.append(edge)
            self.input_edges[copy_node] = [edge]

        self.edges += split_outputs
        self.output_edges[self.start] = split_outputs

        # specail case for the graph has only a variable
        if isinstance(self.end, Variable):
            self.end = copy_nodes[0]

    def input_nodes(self, node):
        return list([e.start for e in self.input_edges[node]])

    def output_nodes(self, node):
        return list([e.end for e in self.output_edges[node]])

    def get_inputs(self, node):
        """Returns the input data for a node.
        """
        return [e.data for e in self.input_edges[node]]

    def get_outputs(self, node):
        """Returns the output data for a node.
        """
        return [e.data for e in self.output_edges[node]]

    def write_outputs(self, node, outputs):
        """Returns the output data for a node.
        """
        if not isinstance(outputs, LinOp.MultOutput):
            outputs = [outputs]
        for e, output in zip(self.output_edges[node], outputs):
            e.data = output

    def write_inputs(self, node, inputs):
        """Returns the input data for a node.
        """
        if not isinstance(inputs, LinOp.MultOutput):
            inputs = [inputs]
        for e, input in zip(self.input_edges[node], inputs):
            e.data = input

    def forward(self, *values, return_list=False):
        """Evaluates the forward composition.
        """
        global y
        y = None

        def forward_eval(node):
            if node == self.start: inputs = list(values)
            else: inputs = self.get_inputs(node)
            if len(inputs) > 1:
                outputs = node.forward(*inputs)
            elif len(inputs) == 1:
                outputs = node.forward(inputs[0])
            else:
                outputs = node.forward()
            if node == self.end: global y; y = outputs
            else: self.write_outputs(node, outputs)

        self.traverse_graph(forward_eval, forward=True)

        if return_list and y is not None and not isinstance(y, LinOp.MultOutput):
            y = [y]
        return y

    def adjoint(self, *values, return_list=False):
        """Evaluates the adjoint composition.
        """
        global y
        y = None

        def adjoint_eval(node):
            if node == self.end: outputs = list(values)
            else: outputs = self.get_outputs(node)
            if len(outputs) > 1:
                inputs = node.adjoint(*outputs)
            elif len(outputs) == 1:
                inputs = node.adjoint(outputs[0])
            else:
                inputs = node.adjoint()
            if node == self.start: global y; y = inputs
            else: self.write_inputs(node, inputs)

        self.traverse_graph(adjoint_eval, forward=False)

        if return_list and y is not None and not isinstance(y, LinOp.MultOutput):
            y = [y]
        return y

    def traverse_graph(self, node_fn, forward):
        """Traverse the graph and apply the given function at each node.

           forward: Traverse in standard or reverse order?
           node_fn: Function to evaluate on each node.
        """
        ready = []
        eval_map = defaultdict(int)
        if forward:
            ready.append(self.start)
            # Constant nodes are leaves as well.
            ready += self.constants
        else:
            ready.append(self.end)

        while len(ready) > 0:
            # Evaluate the given function on curr.
            curr = ready.pop()
            node_fn(curr)

            # Add node that are ready
            # If each input has visited the node, it is ready.

            eval_map[curr] += 1
            if forward: child_edges = self.output_edges.get(curr, [])
            else: child_edges = self.input_edges.get(curr, [])

            for edge in child_edges:
                if forward: node = edge.end
                else: node = edge.start

                eval_map[node] += 1
                if forward: node_inputs_count = len(self.input_edges[node])
                else: node_inputs_count = len(self.output_edges[node])

                if (eval_map[node] == node_inputs_count):
                    ready.append(node)

    def norm_bound(self, final_output_mags):
        """Returns fast upper bound on ||K||.

        Parameters
        ----------
        final_output_mags : list
            Place to store final output magnitudes.
        """
        def node_norm_bound(node):
            # Read input magnitudes off edges.
            if node is self.start:
                input_mags = [1]
            else:
                input_mags = [e.mag for e in self.input_edges[node]]

            # If a node doesn't support norm_bound, propagate that.
            if NotImplemented in input_mags:
                output_mag = NotImplemented
            else:
                output_mag = node.norm_bound(input_mags)

            if node is self.end:
                final_output_mags[0] = output_mag
            else:
                for idx, e in enumerate(self.output_edges[node]):
                    e.mag = output_mag

        self.traverse_graph(node_norm_bound, True)

    def visualize(self, save_path=None):
        """ Visualize the graph with graphviz
        """
        import queue

        import graphviz
        from IPython.display import display
        dot = graphviz.Digraph()
        nodes = {}

        def node_name(obj):
            if not obj in nodes:
                nodes[obj] = 'N%d' % len(nodes)
            return nodes[obj]
        Q = queue.Queue()
        Q.put(self.end)
        while not Q.empty():
            node = Q.get()
            if node not in nodes:
                dot.node(node_name(node), str(node))
            for e in self.input_edges.get(node, []):
                Q.put(e.start)
        for node in nodes.keys():
            for e in self.input_edges.get(node, []):
                dot.edge(node_name(e.start), node_name(e.end))

        if save_path is None:
            display(dot)

    def sanity_check(self, eps=1e-5):
        """ Perform dot product test to check the sanity of this linear operator
        """
        from scipy.misc import face
        m = torch.from_numpy(face().copy()).float().cuda() / 255
        m = m.permute(2, 0, 1).unsqueeze(0)
        d = self.forward(m)

        if isinstance(d, LinOp.MultOutput):
            d2 = [torch.rand_like(e) for e in d]
            m2 = self.adjoint(*d2)
            sum_m = torch.sum(m * m2)
            sum_d = sum([torch.sum(e1 * e2) for e1, e2 in zip(d, d2)])
            diff = torch.abs(sum_m - sum_d)
            rel_diff = torch.abs((sum_m - sum_d) / sum_m)
        else:
            d2 = torch.rand_like(d)
            m2 = self.adjoint(d2)

            sum_m = torch.sum(m * m2)
            sum_d = torch.sum(d * d2)
            diff = torch.abs(sum_m - sum_d)
            rel_diff = torch.abs((sum_m - sum_d) / sum_m)

        if rel_diff < eps:
            print(f'Sanity check passed, diff={diff} rel_diff={rel_diff}')
            return True
        else:
            print(f'Sanity check failed, diff={diff} rel_diff={rel_diff}')
            return False

    def update_vars(self, val):
        """Map sections of val to variables.
        """
        for i, var in enumerate(self.end.variables):
            var.value = val[i]

    def x0(self):
        res = []
        for var in self.end.variables:
            res += [torch.zeros(var.shape)]
        return res

    def __str__(self):
        return self.__class__.__name__


def est_CompGraph_norm(K, tol=1e-3, try_fast_norm=True):
    """Estimates operator norm for L = ||K||.

    Parameters
    ----------
    tol : float
        Accuracy of estimate if not trying for upper bound.
    try_fast_norm : bool
        Whether to try for a fast upper bound.

    Returns
    -------
    float
        Estimate of ||K||.
    """
    if try_fast_norm:
        output_mags = [NotImplemented]
        K.norm_bound(output_mags)
        if NotImplemented not in output_mags:
            return output_mags[0]

    input_data = np.zeros(K.input_size)
    output_data = np.zeros(K.output_size)

    def KtK(x):
        K.forward(x, output_data)
        K.adjoint(output_data, input_data)
        return input_data

    # Define linear operator
    A = LinearOperator((K.input_size, K.input_size),
                       KtK, KtK)

    Knorm = np.sqrt(eigs(A, k=1, M=None, sigma=None, which='LM', tol=tol)[0].real)
    return np.float(Knorm)


def eval(linop, *inputs, zero_out_constant=True):
    G = CompGraph(linop, zero_out_constant)
    if len(inputs) > 1:
        return G.forward(*inputs)
    elif len(inputs) == 1:
        return G.forward(inputs[0])
    else:
        return G.forward()


def adjoint(linop, *inputs, zero_out_constant=True):
    G = CompGraph(linop, zero_out_constant)
    if len(inputs) > 1:
        return G.adjoint(*inputs)
    elif len(inputs) == 1:
        return G.adjoint(inputs[0])
    else:
        return G.adjoint()


def gram(linop, *inputs, zero_out_constant=True):
    K = CompGraph(linop, zero_out_constant)
    outputs = K.forward(*inputs)
    if isinstance(outputs, LinOp.MultOutput):
        return K.adjoint(*outputs)
    return K.adjoint(outputs)
