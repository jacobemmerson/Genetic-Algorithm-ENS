from matplotlib import path

class Gene:
    #the route is a sequential list of integers, where each int is a tractid, and can be used to index into the graph dict
    route: list[int] = []
    def __init__(self):
        self.route = []



#we are more or less stealing our definition of a tract from the gis data def
#if we want sociopolitical income, bring in another dataset and index by tract id (nationally standardized)
class Tract:
    def __init__(self, readable_name: str, geometry:path.Path):
        #FIPS code - left to right - the 2-digit state code, the 3-digit county code, and the 6-digit tract code.
        #self.id: int = id
        self.id: str = readable_name
        self.geometry:path.Path = geometry
        #self.internal_point: tuple[float,float] = internal_point
        #sociopolicitcal factors here


class Zone:
    def __init__(self, GlobalID:str, geometry: path.Path, fzt: str, szt: str):
        self.id: str = GlobalID
        self.geometry:path.Path = geometry
        self.FullZoningType = fzt
        self.ShortZoningType = szt


class Stop:
    def __init__(self, cleverid, displayname, routes, coordinates, tract) -> None:
        self.cleverid:int = cleverid
        self.displayname = displayname
        self.routes:list[str] = routes
        self.coordinates = coordinates
        self.tract = tract

class Segment:
    def __init__(self,route,segIdClever,stopFrom,stopTo,displayName,avgLoad,TripCount):
        self.route = route
        self.segIdClever = segIdClever
        self.stopTo = stopTo
        self.stopFrom = stopFrom
        self.displayName = displayName
        self.avgLoad = avgLoad
        self.TripCount = TripCount
        #service period
        #group
        #direction

"""class Line:
    def __init__(self,seg_indexes,source,dests,drawing_segs):
        self.seg_indexes = seg_indexes
        self.source = source
        self.dest = dests
        self.drawing_segs = drawing_segs"""
class Line:
    def __init__(self, name,DOW,geometry) -> None:
        self.name = name
        self.DOW = DOW
        self.geometry = geometry
        self.stops = []
        self.start = None
        self.end = None
        self.freq = 1
        self.unique_stops = set()
        pass
