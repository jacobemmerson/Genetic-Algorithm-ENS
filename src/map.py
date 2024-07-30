from lib import Tract
import json
from matplotlib import path
import matplotlib as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from collections.abc import Iterable
import collections
from itertools import groupby
import random

from matplotlib import collections as mc

def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

def chunk(list,n):
    return [list[i * n:(i + 1) * n] for i in range((len(list) + n - 1) // n )]


def pull_tracts(census_tracts_csv: str) -> dict[str,Tract]:
    data = json.load(open(census_tracts_csv, "r"))
    
    tracts = []
    for tract in data["features"]:
        #FIPS_code = str(tract["properties"]["statefp"]) + str(tract["properties"]["countyfp"]) + str(tract["properties"]["tractce"])
        
        readable_name = str(tract["properties"]["namelsad"]).replace("Census Tract ","")
        coordinates = tract["geometry"]["coordinates"][0]
        polygon = path.Path(coordinates)
        midpoint = (float(tract["properties"]["intptlat"]),float(tract["properties"]["intptlon"]))
        tracts.append(Tract(readable_name, polygon,midpoint))

    return {t.id: t for t in tracts}



class Map:
    def __init__(self, weight_function, census_tract_path, segments_path):
        self.weight_func = weight_function
        #build tracts from data file at Map construction time
        self._tracts: dict[str,Tract] = pull_tracts(census_tract_path)
        self.graph: dict[str,list[str]] = tracts_to_graph(self._tracts)
        #for a first approximation, graph edge weights will be polygon midpoint distances, which will then be used as an input to the later edge weight function
        self._weights: dict[tuple[str,str],float] = {}
        for node,edges in self.graph.items():
            for e in edges:
                self._weights[(node,e)] = abs((self._tracts[e].internal_point[1]-self._tracts[node].internal_point[1]) / (self._tracts[e].internal_point[0]-self._tracts[node].internal_point[0]))
                self._weights[(e,node)] = self._weights[(node,e)]

        #set of current day transit routes, represented as the sequence of census tracts they pass through
        self._routes: dict[str,int] = {}

        #pull current routes from passed in args path
        self.pull_current_routes(segments_path)

    def pull_current_stops():
        print()

    #visualization of our graph representation
    def visualize_graph(self, interactive: bool):
        g = nx.Graph(self.graph)
        fig = mp.pyplot.figure()
        nx.draw(g, ax=fig.add_subplot())
        if not interactive: 
            # Save plot to file
            mp.use("Agg") 
            plt.close(fig)
            fig.savefig("graph.png")
        else:
            # Display interactive viewer
            mp.pyplot.show()

    #visualization of the real world layout of census tracts and the routes on them
    def visualize_real_world(self):
        fig, ax = plt.subplots()
        for sub in self._tracts:
            patch = patches.PathPatch(self._tracts[sub].geometry, facecolor='orange', lw=0.75)
            ax.add_patch(patch)

        for line,data in self.lines.items():
            for path in data:
                ax.add_patch(patches.PathPatch(path, fill = False, lw = 2.5))

        #for line,data in self.lines.items():
        #    print(line)
        #    ax.add_patch(patches.PathPatch(path.Path(data),facecolor=None, ec = (random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)), fill = False, lw = 2.5))

        #ax.add_patch(patches.PathPatch(path.Path(self.poly_line),facecolor=None, ec = "Blue",fill = False, lw=2.5))

        ax.set_xlim(-80.12, -79.85)
        ax.set_ylim(40.35, 40.51)
        plt.show()
    

    def get_weight(self, node: tuple[str,str]) -> float:
        return self.weight_func(self._weights[node],node)
    
    def pull_current_routes(self,segments_geojson: str)->dict[str,list[tuple[float,float]]]:
        data = json.load(open(segments_geojson, "r"))

        #lines = {}

        #for seg in data["features"]:
        #    if seg["properties"]["ROUTE"] is not None and seg["geometry"] is not None and seg["geometry"]["coordinates"] is not None and seg["properties"]["service_period"] == "Weekday" and seg["properties"]["Group_"] == "Midday" and seg["properties"]["direction_name"]=="INBOUND":
        #        #lines.setdefault(seg["properties"]["ROUTE"],[]).append(chunk(list(flatten(seg["geometry"]["coordinates"])),2))
        #        lines.setdefault(seg["properties"]["ROUTE"],[]).append(seg["geometry"]["coordinates"])

        #self.lines = {k:chunk(list(flatten(v)),2) for (k,v) in lines.items()}


        lines: dict[str, list[path.Path]] = {}
        for seg in data["features"]:
            if seg["properties"]["ROUTE"] is not None and seg["geometry"] is not None and seg["geometry"]["coordinates"] is not None and seg["properties"]["service_period"] == "Weekday" and seg["properties"]["Group_"] == "Midday" and seg["properties"]["direction_name"]=="INBOUND":
                lines.setdefault(seg["properties"]["ROUTE"],[]).append(path.Path(chunk(list(flatten(seg["geometry"]["coordinates"])),2)))

        self.lines = lines



def tracts_to_graph(t:dict[str,Tract] )->dict[str,list[str]]:
    graph = {}
    for k,v in t.items():
        neighbors = []
        for k2,v2 in t.items():
            if v2.geometry.intersects_path(v.geometry) and k2!=k:
                neighbors.append(k2)
        graph[k] = neighbors

    return graph

def dummy_weight_func(base_weight:float, nodes: tuple[str,str]):
    return 1.0

    
m = Map(dummy_weight_func,"./data/Census_Tract_2020.geojson","./data/Segments_by_Route_all_DOW_2206.geojson")
#m.visualize(True)
m.visualize_real_world()
m.visualize_graph(False)

#pull_current_routes("./data/Segments_by_Route_all_DOW_2206.geojson")