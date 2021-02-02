import networkx as nx
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import holoviews as hv
from holoviews.operation.datashader import bundle_graph
from holoviews import opts
from bokeh.models import HoverTool

hv.extension('bokeh')

# This is kinda messy

def draw_graph(df, title='', layout='random', k=None, name=None, bundle=True, width=800, height=800, max_iter=50):
    """
    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame que contiene las aristas de la gráfica, por ejemplo:    
        +--------+--------+
        | source | target |
        +--------+--------+
        |  654   |  1686  |
        +--------+--------+
        | 16344  | 16668  |
        +--------+--------+
        |  1233  |  1937  |
        +--------+--------+
        | 11190  | 12556  |
        +--------+--------+
        |  9032  |  2748  |
        +--------+--------+
    
    title: str, default=''
        Título de la gráfica.

    layout: str, default='random'
        Nombre del layout de networkx que se usará en la gráfica, los 
        posibles son ['random', 'spring', 'kamada']
        
    name: str, default=None
        Nombre con el que se guardará un html de la gráfica, si es None 
        no se guardará el html.
        
    bundle: bool, defaul=True
        Si es True regresa un bundled_graph, que añade estilo a la gráfica.
        
    width: int, default=1000
        Ancho en pixeles que tendrá la gráfica.
        
    height: int, default=1000
        Altura en pixeles que tendrá la gráfica.
        
    max_iter: int, default=50
        Máximo número de iteraciones para el algoritmo nx.spring_layout, que 
        calcula las posiciones óptimas de los nodos en la gráfica.   
        
    Returns
    -------
    Gráfica de Holoviews.
    """
    G = nx.DiGraph()
    G.add_edges_from(df.values)
    
    if layout == 'random':
        pos = nx.random_layout(G, seed=42)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=42, iterations=max_iter, k=k)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
    else: 
        raise ValueError('{} no es un layout soportado'.format(layout))

    nodes_labels = sorted(list(pos.keys()))    
    node_indices = nodes_labels
    
    x = [pos[node][0] for node in nodes_labels]
    y = [pos[node][1] for node in nodes_labels]
    
    n = len(nodes_labels)
    colors = hv.Cycle('Category20').values*int(n/20 + 1)
    colors = colors[0:n]
    node_labels = colors
    
    nodes_df = pd.DataFrame(data={'x': x, 'y':y, 'index': node_indices, 'color': colors})
    edges_df = df.copy()
    edges_df.columns = ['start', 'end']
    
    nodes = hv.Nodes(nodes_df)
    graph = hv.Graph((edges_df, nodes), label=title)
    
    hover = HoverTool(tooltips=[('Node', '@index')])
    graph.opts(node_size=5, node_color='color', xaxis=None, yaxis=None,
               edge_line_width=0.5, edge_line_color='gray', tools=[hover],
               directed=True, arrowhead_length=0.01, width=width, height=height)
    
    if bundle:
        graph = bundle_graph(graph)
        
    if name:
        hv.save(graph, name, fmt='html')
    
    return graph

def draw_communities(df_graph, df_groups, title='', name=None, bundle=True, layout='blobs', cluster_std=0.1, centers=None, random_state=42,
                     width=800, height=800, directed=False, labels=False):

    """
    Parameters
    ----------
    df_graph: pandas.DataFrame
        DataFrame que contiene las aristas de la gráfica, por ejemplo:    
        +--------+--------+
        | source | target |
        +--------+--------+
        |  654   |  1686  |
        +--------+--------+
        | 16344  | 16668  |
        +--------+--------+
        |  1233  |  1937  |
        +--------+--------+
        | 11190  | 12556  |
        +--------+--------+
        |  9032  |  2748  |
        +--------+--------+

    df_groups: pandas.DataFrame
        DataFrame que contiene los nodos y a qué grupo pertenecen, por ejemplo:
        +--------+--------+
        |  node  | group  |
        +--------+--------+
        |  654   |    1   |
        +--------+--------+
        | 16344  |    1   |
        +--------+--------+
        |  1233  |    2   |
        +--------+--------+
        | 11190  |    2   |
        +--------+--------+
        |  9032  |    3   |
        +--------+--------+

    title: str, default=''
        Título de la gráfica.

    name: str, default=None
        Nombre con el que se guardará un html de la gráfica, si es None 
        no se guardará el html.

    bundle: bool, defaul=True
        Si es True regresa un bundled_graph, que añade estilo a la gráfica.

    layout: str, default='blobs'
        Nombre del layout con el que se dibuja la gráfica. Los 
        disponibles son ['blobs', 'kamada'].
            
    cluster_std: float, default=0.1
        Desviación estándar de los blobs para el layout de 'blobs'

    centers: array, default=None
        Array con los centros a usar para el layout de 'blobs'

    random_state: int, default=42
        Semilla a usar a la hora de hacer los 'blobs'

    width: int, default=800
        Ancho en pixeles que tendrá la gráfica.

    height: int, default=800
        Altura en pixeles que tendrá la gráfica.

    directed: bool, default=False
        Si es True dibuja una gráfica dirigida con flechas en las aristas.

    labels: bool, default=False
        Si es True dibuja el nombre de los nodos sobre cada uno.

    Returns
    -------
    Gráfica de Holoviews.
    """

    G = nx.DiGraph()
    G.add_edges_from(df_graph.values)
        
    df_groups.columns = ['node', 'group']
    df_groups = df_groups.sort_values(by='group')

    if layout == 'blobs':
        nodes = df_groups['node'].tolist()           
        # The bigger the community the bigger the cluster_std
        cluster_std_array = cluster_std*np.log(df_groups['group'].value_counts(sort=True))
        X, y = make_blobs(n_samples=df_groups['group'].value_counts(sort=True), centers=centers,
                          shuffle=False, random_state=random_state, cluster_std=cluster_std_array)
        pos = {node: x for node, x in zip(nodes, X)}
        
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(G)
        
    else:
        raise ValueError('{} no es un layout soportado'.format(layout))

    nodes_labels = list(pos.keys())

    x = [pos[node][0] for node in nodes_labels]
    y = [pos[node][1] for node in nodes_labels]

    # Para darle un color único a cada comunidad
    np.random.seed(42)
    n = df_groups['group'].unique().size
    rgb_colors = np.random.randint(low=0, high=255, size=(n, 3))      
    hex_colors = ['#%02x%02x%02x' % tuple(x) for x in rgb_colors]
    dict_colors = dict(zip(df_groups['group'].unique(), hex_colors))
    colors = [dict_colors[group] for group in df_groups['group']]

    # DataFrame que nos ayudará a contruir la gŕafica
    nodes_df = pd.DataFrame(data={'x': x, 'y':y, 'index': nodes_labels, 
                                  'label': nodes_labels, 'color': colors, 
                                  'group': df_groups['group']})
    edges_df = df_graph.copy()
    edges_df.columns = ['start', 'end']

    nodes = hv.Nodes(nodes_df)
    graph = hv.Graph((edges_df, nodes), label=title)

    hover = HoverTool(tooltips=[('Node', '@label'), ('Group', '@group')])
    graph.opts(node_size=5, node_color='color', xaxis=None, yaxis=None,
               edge_line_width=1, edge_line_color='gray', tools=[hover],
               directed=directed, arrowhead_length=0.05, width=width, height=height)    

    if bundle:
        graph = bundle_graph(graph)

    if labels:
        nodes_labels = hv.Labels(graph.nodes.data, ['x', 'y'], 'label')
        graph = (graph * nodes_labels.opts(text_font_size='8pt', text_color='black', 
                                           bgcolor='white', text_font_style='bold'))
    if name:
        hv.save(graph, name, fmt='html')

    return graph


def top_communities(df_graph, df_groups, k=10):
    """
    Regresa las k comunidades con más miembros.

    Parameters
    ----------    
    df_graph: pandas.DataFrame
            DataFrame que contiene las aristas de la gráfica, por ejemplo:    
            +--------+--------+
            | source | target |
            +--------+--------+
            |  654   |  1686  |
            +--------+--------+
            | 16344  | 16668  |
            +--------+--------+
            |  1233  |  1937  |
            +--------+--------+
            | 11190  | 12556  |
            +--------+--------+
            |  9032  |  2748  |
            +--------+--------+

    df_groups: pandas.DataFrame
        DataFrame que contiene los nodos y a qué grupo pertenecen, por ejemplo:
        +--------+--------+
        |  node  | group  |
        +--------+--------+
        |  654   |    1   |
        +--------+--------+
        | 16344  |    1   |
        +--------+--------+
        |  1233  |    2   |
        +--------+--------+
        | 11190  |    2   |
        +--------+--------+
        |  9032  |    3   |
        +--------+--------+

    k: int, default=10
       Número de comunidades a regresar.

    Returns
    -------
    DataFrames con los elementos que pertenecen a las k comunidades con más miembros. 
    """
    df_groups.columns = ['node', 'group']
    df_groups = df_groups.sort_values(by='group')
    
    # Ids de las comunidades con más miembros
    top = df_groups['group'].value_counts(sort=True).index[0:k]
    
    dict_nodes = dict(zip(df_groups['node'], df_groups['group']))   
    from_nodes = df_graph['FromNodeId'].apply(lambda x: dict_nodes.get(x)).isin(top)
    to_nodes = df_graph['ToNodeId'].apply(lambda x: dict_nodes.get(x)).isin(top)
    
    # Nos quedamos únicamente con aquellos que están en las top k comunidades
    df_groups = df_groups[df_groups['group'].isin(top)]
    df_graph = df_graph[from_nodes & to_nodes]
    
    return df_graph, df_groups