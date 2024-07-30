#this file is designed to build our bus stop map which will be used as the geospatial ground truth for network decisions

import json
import codecs

import numpy as np
from numpy import dtype

class Map:

    def __init__(self):
        #data = json.load(codecs.open('sample.json', 'r', 'utf-8-sig'))
        data = json.load(open("./data/PRT_Stops_Current_2402.geojson", "r"))
        #print(data)


        #define the record type
        #stop_dtype = dtype([("FID",np.intc),("lat",np.double),("lon",np.double),("lines",'O'),("ctract",np.longlong)])
        stop_dtype = dtype([("FID",np.intc),("coord",np.double,2),("lines",'O'),("ctract",np.longlong)])        
        print("dtype is:" + str(stop_dtype))


        #flatten the data a little bit
        data = [line["properties"] for line in data["features"]]
        print(str(data[2500]))



        #line_str_to_int = {}
        #parse into a standard python list
        #parsed = [(int(stop["FID"]),float(stop["Latitude"]),float(stop["Longitude"]),str(stop["Routes_ser"]),0) for stop in data]
        parsed = []
        for stop in data:
            lines = stop["Routes_ser"].split(",")
            #for l in lines:
                #ident = line_str_to_int.setdefault(l,len(line_str_to_int))
            parsed.append((int(stop["FID"]),(float(stop["Latitude"]),float(stop["Longitude"])),lines,0))

        #int_to_line_str = {v: k for k, v in line_str_to_int.items()}
        
        #print("post parse line 2500:" +str(parsed[2500]) + "routes served " + int_to_line_str[parsed[2500][3]])
        print("post parse line 2500:" +str(parsed[2500]) + "routes served " + str(parsed[2500][3]))


        #create a numpy array
        map = np.rec.array(parsed,dtype=stop_dtype)
        print(map)

        print("FUCKYOU:"+str(map[2500]["lines"]))

        sorted = map.sort(order="FID")
        print(sorted)

Map()