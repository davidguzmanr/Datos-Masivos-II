import pandas as pd
import numpy as np

import holoviews as hv
import networkx as nx
from holoviews import opts
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import HoverTool

hv.extension('bokeh')
defaults = dict(width=800, height=800)
hv.opts.defaults(opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))


def directed_graph(df, name=None, title=''):
    """
    Regresa una grafo dirigido para ver los patrones de las compras.
    
    df: pandas.DataFrame
        Dataframe con los patrones frecuentes, por ejemplo:
        +------------------------------------+---------+
        |               items                | support |
        +------------------------------------+---------+
        | ('whole milk', 'other vegetables') |  0.015  |
        +------------------------------------+---------+
        |    ('whole milk', 'rolls/buns')    |  0.014  |
        +------------------------------------+---------+
        |       ('whole milk', 'soda')       |  0.012  |
        +------------------------------------+---------+
        |      ('whole milk', 'yogurt')      |  0.011  |
        +------------------------------------+---------+
        | ('other vegetables', 'rolls/buns') |  0.011  |
        +------------------------------------+---------+
        
    name: str, default=None
        Nombre con el que se guardará un html de la gráfica, si es 
        None no se guardará el html.

    title: str, default=''
        Título para la gráfica
    """    
    
    source = [x[0] for x in df['items']]
    target = [x[1] for x in df['items']]
    edge_weights = df['support'].values
    
    graph = hv.Graph(((source, target, ),))

    weights_data = pd.DataFrame(columns=['x', 'y', 'weight'])
    for (edge,weight) in zip(graph.edgepaths.data, edge_weights):
        mid_point = (edge[0] + edge[1])/2
        weights_data.loc[len(weights_data)] = [mid_point[0], mid_point[1], round(weight,4)]

    
    nodes_labels = hv.Labels(graph.nodes.data, ['x', 'y'], 'index')
    edges_labels = hv.Labels(weights_data, ['x', 'y'], 'weight')

    hover = HoverTool(tooltips=[('Item', '@index_hover')])
    graph.opts(edge_line_width=3, edge_line_color='black', edge_alpha=0.3,
               node_color='#33D1FF', node_line_color='black', node_size=50,
               directed=True, arrowhead_length=0.04, node_alpha=0.5, 
               edge_hover_line_color='#32E319', node_hover_color='#32E319',
               title=title, tools=[hover])

    graph = (graph * nodes_labels.opts(text_font_size='8pt', text_color='black', 
                                       bgcolor='white', text_font_style='bold'))
    graph = (graph * edges_labels.opts(text_font_size='8pt', text_color='black', 
                                       bgcolor='white', text_font_style='bold'))
    
    if name:
        hv.save(graph, name, fmt='html')    
        
    return graph


def weighted_graph(df, name=None, title=''):
    """
    Regresa una grafo donde el grosor de las aristas es proporcional
    a su peso.
    
    df: pandas.DataFrame
        Dataframe con los patrones frecuentes, por ejemplo:
        +------------------------------------+---------+
        |               items                | support |
        +------------------------------------+---------+
        | ('whole milk', 'other vegetables') |  0.015  |
        +------------------------------------+---------+
        |    ('whole milk', 'rolls/buns')    |  0.014  |
        +------------------------------------+---------+
        |       ('whole milk', 'soda')       |  0.012  |
        +------------------------------------+---------+
        |      ('whole milk', 'yogurt')      |  0.011  |
        +------------------------------------+---------+
        | ('other vegetables', 'rolls/buns') |  0.011  |
        +------------------------------------+---------+
        
    name: str, default=None
        Nombre con el que se guardará un html de la gráfica, si es 
        None no se guardará el html.

    title: str, default=''
        Título para la gráfica
    """
    
    source = [x[0] for x in df['items']]
    target = [x[1] for x in df['items']]
    edge_weights = df['support'].values

    graph = hv.Graph(((source, target, edge_weights),))

    node_labels = list(graph.nodes.data['index'])
    node_indices = np.arange(len(node_labels))

    x, y = graph.nodes.array([0, 1]).T

    data = graph.nodes.data
    dict_coordinates = dict(zip(data['index'], data[['x','y']].values))

    # Compute edge paths
    def bezier(start, end, control, steps=np.linspace(0, 1, 100)):
        return (1-steps)**2*start + 2*(1-steps)*steps*control+steps**2*end

    paths = []
    for (start,end) in zip(source, target):
        start_x, start_y = dict_coordinates[start]
        end_x, end_y = dict_coordinates[end]
        paths.append(np.column_stack([bezier(start_x, end_x, 0), bezier(start_y, end_y, 0)]))

    nodes = hv.Nodes((x, y, node_indices, node_labels), vdims='Type')
    graph = hv.Graph(((source, target, 200*edge_weights), None, paths), vdims='Weight')

    hover = HoverTool(tooltips=[('Item', '@index_hover')])
    graph.opts(edge_line_width='Weight', edge_line_color='black', edge_alpha=0.3,
               node_color='#33D1FF', node_line_color='black', node_size=50,
               directed=True, arrowhead_length=0.04, node_alpha=0.5, 
               edge_hover_line_color='#32E319', node_hover_color='#32E319',
               title=title, tools=[hover])
    
    weights_data = pd.DataFrame(columns=['x', 'y', 'weight'])
    for (path,weight) in zip(paths, edge_weights):
        mid_point = path[int(len(path)/2)]
        weights_data.loc[len(weights_data)] = [mid_point[0], mid_point[1], round(weight,4)]

    nodes_labels = hv.Labels(graph.nodes, ['x', 'y'], 'index')
    edges_labels = hv.Labels(weights_data, ['x', 'y'], 'weight')

    graph = (graph * nodes_labels.opts(text_font_size='8pt', text_color='black', 
                                              bgcolor='white', text_font_style='bold'))
    graph = (graph * edges_labels.opts(text_font_size='8pt', text_color='black', 
                                       bgcolor='white', text_font_style='bold'))
    
    if name:
        hv.save(graph, name, fmt='html') 
        
    return graph