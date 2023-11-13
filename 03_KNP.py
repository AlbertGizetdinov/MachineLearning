import random
import sys

import matplotlib.pyplot as plt
import networkx as nx


def generate_data(node_amount):
    edge_data = []

    for i_node in range(node_amount):
        for j_node in range(i_node + 1, node_amount):
            edge_data.append([i_node, j_node, random.randint(1, 10)])

    return edge_data


def find_edge_with_weight_predicate(graph_data, with_min_weight):
    edge_data = None
    weight_predicate = sys.maxsize if with_min_weight else 0

    for i_node, j_node, weight in graph_data:
        if ((weight < weight_predicate and with_min_weight)
                or (weight > weight_predicate and not with_min_weight)):
            weight_predicate = weight
            edge_data = [i_node, j_node, weight]

    return edge_data


def find_nearest_edge(connected_nodes, graph_data):
    edge_data = None
    min_weight = sys.maxsize

    for connected_node in connected_nodes:
        for i_node, j_node, weight in graph_data:
            i_node_can_connect_to_j_node = i_node == connected_node and j_node not in connected_nodes
            j_node_can_connect_to_i_node = j_node == connected_node and i_node not in connected_nodes

            if (i_node_can_connect_to_j_node or j_node_can_connect_to_i_node) and weight < min_weight:
                min_weight = weight
                edge_data = [i_node, j_node, weight]

    return edge_data


def find_nearest_edges(starting_edge, initial_graph_data):
    graph_data = initial_graph_data.copy()
    connected_nodes = set(starting_edge[:2])
    nearest_edges = [starting_edge]

    graph_data.remove(starting_edge)

    while len(connected_nodes) != node_amount:
        nearest_edge = find_nearest_edge(connected_nodes, graph_data)

        nearest_edges.append(nearest_edge)
        connected_nodes.update(nearest_edge[:2])
        graph_data.remove(nearest_edge)

    return nearest_edges


def split_into_clusters(nearest_edges, cluster_amount):
    cluster_edges = nearest_edges.copy()

    for i in range(cluster_amount - 1):
        longest_edge = find_edge_with_weight_predicate(cluster_edges, with_min_weight=False)
        cluster_edges.remove(longest_edge)

    return cluster_edges


def display_graph(node_amount, edges, graph_data):
    G = nx.Graph()
    G.add_nodes_from(range(node_amount))

    missing_edges = [edge for edge in graph_data if edge not in edges]

    for i_node, j_node, weight in edges:
        G.add_edge(i_node, j_node, weight=weight, color='g')

    for i_node, j_node, weight in missing_edges:
        G.add_edge(i_node, j_node, weight=weight, color='r')

    pos = nx.spring_layout(G, seed=1)
    weights = nx.get_edge_attributes(G, 'weight')

    nx.draw(
        G,
        pos,
        edge_color=[G[u][v]['color'] for u, v in G.edges()],
        width=[4 if G[u][v]['color'] == 'g' else 1 for u, v in G.edges()],
        node_color='g'
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.show()


if __name__ == '__main__':
    G = nx.Graph()

    cluster_amount = 2
    node_amount = 7

    graph_data = generate_data(node_amount)
    starting_edge = find_edge_with_weight_predicate(graph_data, with_min_weight=True)

    nearest_edges = find_nearest_edges(starting_edge, graph_data)
    display_graph(node_amount, nearest_edges, graph_data)

    cluster_edges = split_into_clusters(nearest_edges, cluster_amount)
    display_graph(node_amount, cluster_edges, graph_data)
