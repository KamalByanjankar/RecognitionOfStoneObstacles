from glob import glob
from file_helper import get_time_domain_data_without_offset, load_data_points
from filter import apply_filter_data_set
from echo_finder import get_echos
from fft_implementation import get_fft_from_data_set
from utility import get_distance
from plot_graph import plot_graphs

BASE_FILE_PATH = './rawData/dataSet'

"""
load data and calculates information
i.e 
time-domain-data
removes-offset
initial-filtered-data
calculate-distance from time-of-flight
filtered-data, after removal of insufficient distance from time-of-flight
fft-data
"""
class LoadData:
    
    """ 
    objectName is the name of object, eg: bigCobbleStoneFlat
    distance is the distance folder,
    it can be distance=90, distance=[90,100,120], distance='ALL' or 'all' or '*'
    
    """
    def __init__(self, objectName = 'bigCobbleStoneFlat', distance = 80):
        self.objectName = objectName
        self.distance = distance
        # finally self.filenames has all the files that are requested on load
        self.__load_files()
        
        # calculate everything time-domain, fft, filtered-data, time-of-flight
        self.__set_data()
        
    
    def __load_files(self):
        if type(self.distance) == int:
            self.filenames = glob('{}//{}//1//{}.csv'.format(BASE_FILE_PATH, self.objectName, self.distance))
        elif type(self.distance) == list:
            self.filenames = []
            for dist in self.distance:
                self.filenames.append('{}//{}//1//{}.csv'.format(BASE_FILE_PATH, self.objectName, dist))
        elif self.distance == 'ALL' or self.distance == '*' or self.distance == 'all':
            self.filenames = glob('{}//{}//1//{}.csv'.format(BASE_FILE_PATH, self.objectName, '*'), recursive=True)
            
    def __set_data(self):
        self.data = []
        for filename in self.filenames:
            row = {}
            row['distance'] = int(filename.split('\\')[-1].split('/')[-1].split('.csv')[0])
            row['timeDomain'] = load_data_points(filename)
            row['timeDomainWithoutOffset'] = get_time_domain_data_without_offset(row['timeDomain'])
            row['initialFilteredData'] = apply_filter_data_set(row['timeDomainWithoutOffset'])
            # update row by calculating last distance from initalFilteredData
#             row['actualDistance'] = get_distance(row['initialFilteredData'][-1])
            row['echo'] = get_echos(row['initialFilteredData'])
            row['fft'] = get_fft_from_data_set(row['echo'])
            self.data.append(row)
        self.data = sorted(self.data, key = lambda i: (i['distance']))
        