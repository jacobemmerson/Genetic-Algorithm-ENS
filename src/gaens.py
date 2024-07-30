from .lib import Tract,Zone,Stop,Segment,Line
import json
import csv
from matplotlib import path 
import matplotlib as mp
import numpy as np
from numpy import ndarray
from matplotlib import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import collections  as mc
from haversine import haversine, Unit
from .pop import Population
import math

######################stackoverflow containment zone############################
def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z

def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)

def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)

def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)

def distance(p0,p1):
    return length(vector(p0,p1))

def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)

def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest 
# distance from pnt to the line and the coordinates of the 
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line. 
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
    t = dot(line_unitvec, pnt_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, nearest)
#-----------------

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
        #readable_name = str(tract["properties"]["namelsad"]).replace("Census Tract ","").split(".")[0]
        id = tract["properties"]["NAME"]
        coordinates = tract["geometry"]["coordinates"][0][0]
        polygon = path.Path(coordinates)
        #midpoint = (float(tract["properties"]["intptlat"]),float(tract["properties"]["intptlon"]))
        tracts.append(Tract(id, polygon))

    return {t.id: t for t in tracts}

def pull_zoning(zoning_zones_json: str) -> dict[str,Zone]:
    data = json.load(open(zoning_zones_json, "r"))

    zones = []
    for zone in data["features"]:
        id = zone["properties"]["GlobalID"]
        #FIXME THIS IS A HACK!!!!!
        if zone["geometry"] is None or zone["geometry"]["type"] == "MultiPolygon":
            continue
        coordinates = zone["geometry"]["coordinates"][0]
        #print(str(zone["properties"]["OBJECTID"]  ) +" "+ str(len(coordinates)))
        polygon = path.Path(coordinates)
        fullzone = zone["properties"]["full_zoning_type"]
        shortzone = zone["properties"]["legendtype"]
        zones.append(Zone(id,polygon,fullzone,shortzone))

    return {t.id: t for t in zones}



def pull_census_data(census_raw_data_csv: str, census_shapes) -> np.array:
        #np array
        #census_dtype = np.dtype([("tract_id",np.intc),("type",('U',10)),("median_age",np.double),("demographics",[("white",np.double),("black",np.double),("asian",np.double)]),("median_income",np.intc),("owned_houses",np.double)])
        demographics_dtype = np.dtype([("black",np.double),("asian",np.double),("white",np.double)])
        #census_dtype = np.dtype([("tract_id",int),("type",('U',10)),("median_age",np.double),("demographics",demographics_dtype),("median_income",int),("owned_houses",np.double)])
        #FIXME: this is fucking ridiculous
        census_dtype = np.dtype([("tract_id",np.object_),("type",np.object_),("median_age",np.object_),("demographics",demographics_dtype),("median_income",np.object_),("owned_houses",np.object_)])

        with open(census_raw_data_csv, newline='') as csvfile:
            spamreader = csv.reader(csvfile)
            csv_data = [x for x in spamreader]
            #print(csv_data)

        census_data = []
        for idx,col in enumerate(csv_data[0]):
            if idx == 0:
                continue
            split = col.split("_")
            if len(split) <=1:
                continue
            id = split[0]
            if "." in id:
                continue
            if not id in census_shapes:
                #print("no census shape for id: " + str(id) + " skipping adding it to the set") 
                continue
            tract_type = split[1]
            median_age = csv_data[1][idx]
            try:
                median_age = float(median_age)
            except ValueError:
                median_age = None
            white = csv_data[2][idx].replace("%","")
            try:
                white = float(white)
            except ValueError:
                white = None
            black = csv_data[3][idx].replace("%","")
            try:
                black = float(black)
            except ValueError:
                black = None
            asian = csv_data[4][idx].replace("%","")
            try:
                asian = float(asian)
            except ValueError:
                asian = None
            median_income = csv_data[5][idx].replace(",","")
            try:
                median_income = int(median_income)
            except ValueError:
                median_income = None
            ownership = csv_data[6][idx].replace("%","")
            try:
                ownership = float(ownership)
            except ValueError:
                ownership = None

            census_data.append((int(id),tract_type,median_age,(white,black,asian),median_income,ownership))
        
        #print("we pulled data for " + str(len(census_data)) + " tracts")

        #I genuinely do not think it is worth it to get a pointer into the shapes array. just index by the shared key (id)
        return np.array(census_data, dtype = census_dtype)



def pull_equity_scores(equity_score:str):
    data = json.load(open(equity_score, "r"))

    equity_scores = {}

    for tract in data["features"]:
        tractce = tract["properties"]["TRACTCE"]
        #if str(tractce)[-1] == "0" and str(tractce)[-2] == "0":
        #    tractce = tractce[:-2]
        #else:
        #    continue
        Sc_lowincH = tract["properties"]["Sc_lowincH"]
        Sc_LowWage = tract["properties"]["Sc_LowWage"]
        sc_Access = tract["properties"]["sc_Access_"]
        Sco_Disabi = tract["properties"]["Sco_Disabi"]
        Sco_NoVehi = tract["properties"]["Sco_NoVehi"]
        Sco_LEP = tract["properties"]["Sco_LEP"]
        Sco_Minori = tract["properties"]["Sco_Minori"]
        Sco_over65 = tract["properties"]["Sco_over65"]
        Sco_under1 = tract["properties"]["Sco_under1"]
        Sco_femHH = tract["properties"]["Sco_femHH"]
        Final8_Ind = tract["properties"]["Final8_Ind"]
        poly = tract['geometry']['coordinates']

        equity_scores[tractce] = (Sc_lowincH,Sc_LowWage,sc_Access,Sco_Disabi,Sco_NoVehi,Sco_LEP,Sco_Minori,Sco_over65,Sco_under1,Sco_femHH,Final8_Ind,poly)

    return equity_scores

class GAES:
    def __init__(self,census_shapes_json,census_transport_data_csv):
        #compile time
        #--------build blob data structures------
        #-------------level 1 blobs--------------
        #--census shapes
        #dict[str: Tract]
        self.census_shapes: dict[str:Tract] = pull_tracts(census_shapes_json)
        print("[init] pulled " + str(len(self.census_shapes)) + " tract shapes")

        #----------------------------------------
        #-------------level 2 blobs--------------
        #--census data blob
        #nparray of type census_dtype (check fn for definition)
        self.census_data = pull_census_data(census_transport_data_csv,self.census_shapes)
        print("[init] pulled data for " + str(len(self.census_data)) + " tracts")

        #--zoning blob
        self.zones = pull_zoning("data/zoning.geojson")
        print("[init] pulled data for " + str(len(self.zones)) + " city zones")

        #--PRT equity scores
        self.equity_scores = pull_equity_scores("data/2018_Equity_Index.geojson")
        print("[init] pulled equity scores for " + str(len(self.equity_scores)) + " tracts")

        #--gmaps blob
        #TODO: make a decision here!
        #----------------------------------------
        #-------------level 3 blobs--------------
        #stops blob
        self.stops = self.pull_stops("data/PRT_Stops_Current_2402.geojson",self.census_shapes)
        print("[init] pulled " + str(len(self.stops)) + " stops")
        print("[init] binned by stop, there are " + str(len(self.stops_by_route.items())) + " bins")
        #----------------------------------------

        #-------------level 3 blobs--------------
        #--segments blob (TRAINING DATASET)
        self.segs = self.pull_segs("data/Segments_by_Route_all_DOW_2206.geojson","/stop_to_demand.json")
        print("[init] pulled " + str(len(self.segs)) + " segments")
        #----------------------------------------

        self.base_lines = self.pull_current_lines("data/PRT_Routes_current_2310.geojson")
        print("[init] pulled " + str(len(self.base_lines)) + " lines")
        self.order_lines()
        print("[init] lines' stop arrays successfully re-ordered")
        

    #dedup function for segments testing equality only on stopFrom and stopTo equality
    def shitass_function(self,l,stopFrom,stopTo):
        for e in l:
            if self.segs[e]['stopFrom']==stopFrom and self.segs[e]['stopTo']==stopTo:
                return True
        return False

    """def pull_current_routes(self)->dict[str, list[int]]:

        #FIXME: this is a harcoded list of routes which I know ahead of time wont yield clean routes
        uniquely_fucked_up = {'O1', '4', '2', '12', 'Y47', '20', '55', 'Y46', '59', '61A', '74', '53L', '61B', 'O5', '22'}

        #dict of str for name to list[index into seg blob]
        lines = {}
        for index,seg in enumerate(self.segs):
            #if seg["properties"]["ROUTE"] is not None and seg["geometry"] is not None and seg["geometry"]["coordinates"] is not None and seg["properties"]["service_period"] == "Weekday" and seg["properties"]["Group_"] == "Midday" and seg["properties"]["direction_name"]=="INBOUND":
            if seg["route"] in uniquely_fucked_up:
                continue
            lines.setdefault(seg['route']+"~"+seg["service_period"]+seg["group"]+seg["direction"],[]).append(index)
            #if not self.shitass_function(lines[seg['route']], seg['stopFrom'],seg['stopTo']):
        #create sorted lines
        final_lines = {}
        for line_name,line in lines.items():                
            #source_id to seg index in segment blob
            sources = {}
            #set of all dests
            dests = set()
            for seg_indx in line:
                sources[self.segs[seg_indx]['stopFrom']] = seg_indx
                dests.add(self.segs[seg_indx]['stopTo'])

            #print(sources)
            #print(dests)

            s = set(sources.keys())
            sources = s.difference(dests)
            dests = dests.difference(s)
            #print("looking at line: " + line_name)
            #print("sources - dests is " + str(sources))
            #print("dests - sources is " + str(dests))
            drawing_segments = [[(self.segs[x]["long_from"],self.segs[x]["lat_from"]),(self.segs[x]["long_to"],self.segs[x]["lat_to"])] for x in line]
            #lc = mc.LineCollection(drawing_segments, linewidths=2)
            #fig, ax = plt.subplots()
            #ax.add_collection(lc)
            #ax.autoscale()
            #ax.margins(0.1)
            #plt.show()

            #index of first segment in segs blob
            #first = None
            #for ele in sources.keys():
            #    if ele in dests:
            #        continue
            #    else:
            #        first = sources[ele]
            #        break
            #print("first is: " + str(first) + " source: " + str(self.segs[first]["stopFrom"]) + " dest: " + str(self.segs[first]["stopTo"]))
            #if first == None:
            #    print("FUCKFUCKFUCKFUCKFUCK")
            #    continue
            #sorted = [first]
            #while len(sorted) != len(line):
            #    #get sink of last ele in sorted
            #    sink = self.segs[sorted[-1]]["stopTo"]
            #    #whos source is this sink?
            #    next = sources[sink]
            #    sorted.append(next)
            #lines[line_name] = sorted
            #print("MADEIT")
            final_lines[line_name] = Line(line,sources,dests,drawing_segments)
        self.lines = final_lines"""
    
    def pull_current_lines(self,lines_json):
        data = json.load(open(lines_json, "r"))
        lines = []

        for line in data["features"]:
            if line['properties']['Mode'] != "Bus": 
                continue
            name = line["properties"]["Route_2"]
            DOW = line["properties"]["DOW"]
            if line["geometry"] is None:
                continue
            geometry = line["geometry"]["coordinates"]
            route = Line(name,DOW,geometry)
            route.stops = self.stops_by_route[name]
            route.freq = line['properties']['AvgDaily_R']
            if route.stops is None:
                print("could not get stops by route for " + str(name))
            lines.append(route)

        return {t.name: t for t in lines}

    def order_lines(self):
        for (name,line) in self.base_lines.items():
            #print("for line " + str(name) + " initial stop ordering is: ",end="")
            #for i in self.base_lines[name].stops:
            #    print(str(i.cleverid) + ", ",end="")
            #print("\n")
            intermediate = []
            for stop in line.stops:
                best_match = None
                shortest = math.inf
                for (indx,piece) in enumerate(line.geometry):
                    #calculate distance from stop coords to this line seg.
                    dist = pnt2line(start=(piece[0][0],piece[0][1],0),end=(piece[1][0],piece[1][1],0),pnt=(stop.coordinates[0],stop.coordinates[1],0))[0]
                    if dist < shortest:
                        shortest = dist
                        best_match = indx
                intermediate.append((stop,best_match))
            sorted_list = sorted(intermediate,key=lambda x: x[1])
            self.base_lines[name].stops = [x[0] for x in sorted_list]
            #print("after sorting line " + str(name) + " new stop ordering is: ",end="")
            #for i in self.base_lines[name].stops:
            #    print(str(i.cleverid) + ", ",end="")
            #print("\n")

    def pull_stops(self,curr_stops_json: str, census_shapes: dict[str, Tract])->dict[str,Stop]:
        data = json.load(open(curr_stops_json, "r"))

        stops = []
        for stop in data["features"]:
            stopid = stop["properties"]["CleverID"]
            displayname = stop["properties"]["StopName"]
            routes = stop["properties"]["Routes_ser"].split(",")
            routes = [x.replace("\n",",").strip() for x in routes]
            #for ele in routes:
            #    ele.replace("\n","")
            #    ele.strip()
            coordinates = stop["geometry"]["coordinates"]
            
            contains = True
            
            for key, value in census_shapes.items():
                if value.geometry.contains_point(coordinates):
                    contains = False
                    tract = key
                    continue

            if contains: continue

            stops.append(Stop(stopid,displayname,routes,coordinates,tract))
        

        #bin stops by route
        self.stops_by_route = {}
        for stop in stops:
            for r in stop.routes:
                self.stops_by_route.setdefault(r,[]).append(stop)

        return {t.cleverid: t for t in stops}

    def pull_segs(self,segs_json:str,demands_json:str)->np.array:
        data = json.load(open(segs_json, "r"))

        #demands = json.load(open(demands_json,"r"))

        #segment dtype - define a segment using only np compliant datatypes
        seg_dtype = np.dtype([('route', np.unicode_, 5),("linkStopId",np.unicode_, 13),("stopFrom",int),("stopTo",int),("displayName",np.unicode_,75),("avgLoad",np.double),("tripCount",int),("service_period",np.unicode_,10),("group",np.unicode_,10),("direction",np.unicode_,10),("lat_from",np.double),("long_from",np.double),("lat_to",np.double),("long_to",np.double),("demand",np.double)])

        segs = []
        for seg in data["features"]:
            route = seg["properties"]["route_name"]
            linkStopId = seg["properties"]["segID_clever"]
            if linkStopId is None:
                continue
            fromto = linkStopId.split("_")
            stopFrom = int(fromto[0])
            stopTo = int(fromto[1])

            #if both sides of the cleverid are not in our stops blob, dont add this segment

            if stopFrom not in self.stops or stopTo not in self.stops:
                continue

            dispName = seg["properties"]["Description"]
            avgLoad = seg["properties"]["avgLoad"]
            TripCount = seg["properties"]["tripCount"]
            service_period = seg["properties"]["service_period"]
            group = seg["properties"]["Group_"]
            direction = seg["properties"]["direction_name"]
            lat_from = seg["properties"]["Lat_from"]
            long_from = seg["properties"]["Long_from"]
            lat_to = seg["properties"]["Lat_to"]
            long_to = seg["properties"]["Long_to"]
            #demand = demands_json[str(stopFrom + "_" + stopTo)]
            segs.append((route,linkStopId,stopFrom,stopTo,dispName,avgLoad,TripCount,service_period,group,direction,lat_from,long_from,lat_to,long_to,0.0))
        np_segs = np.array(segs,dtype = seg_dtype)
        return np_segs


    def visualize_real_world(self):
        fig, ax = plt.subplots()
        for sub in self.census_shapes:
            patch = patches.PathPatch(self.census_shapes[sub].geometry, facecolor='orange', lw=0.75)
            ax.add_patch(patch)

        #for line,data in self.lines.items():
        #    for path in data:
        #        ax.add_patch(patches.PathPatch(path, fill = False, lw = 2.5))

        #for line,data in self.lines.items():
        #    print(line)
        #    ax.add_patch(patches.PathPatch(path.Path(data),facecolor=None, ec = (random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)), fill = False, lw = 2.5))

        #ax.add_patch(patches.PathPatch(path.Path(self.poly_line),facecolor=None, ec = "Blue",fill = False, lw=2.5))

        #ax.set_xlim(-80.12, -79.85)
        #ax.set_ylim(40.35, 40.51)
        ax.set_xlim(-81, -79)
        ax.set_ylim(40, 41)
        plt.show()

    def build_initial_population(self):
        #define the geographic network - FIXME: this takes fucking forever lmao
        network:dict[int:list[int]] = {}
        for k,i in self.stops.items():
            network.setdefault(i.cleverid,[])
            for l,j in self.stops.items():
                #if j is within 0.5km? of this stop, add it to their adjacency list
                if i.cleverid!=j.cleverid and haversine(j.coordinates,i.coordinates) <= 0.5:
                    network[i.cleverid].append(j.cleverid)

        individual = {}
        #define the first individual
        for (name,line) in self.base_lines.items():
            new_arr = []
            for (idx,stop) in enumerate(line.stops):
                if idx == len(line.stops)-1:
                    break
                new_arr.append([line.stops[idx].cleverid,line.stops[idx+1].cleverid])
            
            line.stops = np.array(new_arr)
            line.start = new_arr[0][0]
            line.end = new_arr[-1][-1]
            individual[name] = line 

        first_individual = individual# dictionary of lines with values = line object
        #call the population constructor
        return network,first_individual
        #self.pop = Population(network,first_individual,args=[],segment_demands_path="stop_to_demand.json")

#g = GAES(census_shapes_json="data/2016_census_block.geojson",census_transport_data_csv="data/transport_census.csv")
#pop = g.build_initial_population()
#print(pop[1])
#print(g.pop.get_fitness())
#g.pop.get_fitness()

#g.visualize_real_world()
#print(g.lines)
#for seg in g.lines["71A"]:
#    print(str(seg["stopFrom"]) + " " + str(seg["stopTo"]))
"""fucked = []
for i,l in g.lines.items():
    if (len(l[1]) > 1 or len(l[1])>1) and (g.segs[l[0]][0]["route"] not in fucked):   
        #print("lines are fucked: " + str(fucked))
        print(str(g.segs[l[0]][0]["route"]+"~"+g.segs[l[0]][0]["service_period"]+g.segs[l[0]][0]["group"]+g.segs[l[0]][0]["direction"]) + " has more than one source or sink")
        fucked.append(i)
        #lc_colors = [(245, 40, 145, 0.8)]*len(l[3])
        #lc = mc.LineCollection(l[3], linewidths=2,colors=lc_colors)
        #fig, ax = plt.subplots()
        #ax.add_collection(lc)
        #for sub in g.census_shapes:
        #    patch = patches.PathPatch(g.census_shapes[sub].geometry, facecolor='orange', lw=0.75)
        #    ax.add_patch(patch)
        #ax.set_xlim(-81, -79)
        #ax.set_ylim(40, 41)
        #ax.autoscale()
        #ax.margins(0.1)
        #plt.show()

print(len(fucked))
print(len(g.lines.keys()))

for ele in fucked:
    del g.lines[ele]

replaced = []
not_replaceable = []
for ele in fucked:
    #make sure there a same line of diff d/t/d variation remaining in g.lines
    #fucked_route = g.lines[ele][0]

    fucked_route_id = ele.split("~")[0]
    #print("trying to find " + str(fucked_route_id))
    found = False
    for k in g.lines.keys():
        this_route_id = k.split("~")[0]
        #print("checking " + str(this_route))
        if this_route_id == fucked_route_id:
            #print("found a good replacement")
            found = True

    if not found:
        #print("no replacement found for " + str(ele))
        not_replaceable.append(ele)
    else:
        #print("replacement found for " + str(ele))
        replaced.append(ele)

print("replaceable: " + str(len(replaced)) + " not replaced: " + str(len(not_replaceable)) + " total: " + str(len(replaced) + len(not_replaceable)))

unique = set()
for ele in not_replaceable:
    id = ele.split("~")[0]
    unique.add(id)
print("uniquely fucked up" + str(unique))"""