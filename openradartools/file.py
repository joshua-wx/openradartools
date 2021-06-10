import os
import glob
import zipfile
import tempfile
from datetime import datetime

#Other libs
import pandas
import pyart
import numpy as np

def mkdir(mydir):
    if os.path.exists(mydir):
        return None
    try:
        os.makedirs(mydir)
    except FileExistsError:
        return None
    return None

def remove_path(path):
    cmd = 'rm -rf ' + path
    os.system(cmd)

def unpack_zip(zip_ffn):
    """
    Unpacks zip file in temp directory

    Parameters:
    ===========
        zip_ffn: str
            Full filename to zip file 
            
    Returns:
    ========
        temp_dir: string
            Path to temp directory
    """    
    #build temp dir
    temp_dir = tempfile.mkdtemp()
    #unpack tar
    zip_fd = zipfile.ZipFile(zip_ffn)
    zip_fd.extractall(path=temp_dir)
    zip_fd.close()
    return temp_dir

def pack_zip(zip_fn, zip_path,  ffn_list):
    """
    Packs zip file from file list

    Parameters:
    ===========
        zip_fn: str
            filename of output tar file 
        tar_path: str
            path of output tar file 
        ffn_list: list
            list of ffn to add to tar
    """   
    #create path
    if not os.path.exists(zip_path):
        util.mkdir(zip_path)
    #zip filename
    zip_ffn    = '/'.join([zip_path,zip_fn])
    #remove if it exists
    if os.path.isfile(zip_ffn):
        os.system('rm -f ' + zip_ffn)
    #build zip path
    zip_cmd = 'zip -jq1'
    files_to_zip = ' '.join(ffn_list)
    cmd = ' '.join([zip_cmd, zip_ffn, files_to_zip])
    os.system(cmd)
    return None

def findin_sitelist(config_dict, radar_id, radar_dt):    
       
    id_list  = config_dict['id']
    dts_list = config_dict['postchange_start']
    #replace empty values with 1900
    for i, dts in enumerate(dts_list):
        if dts == '-':
            dts_list[i] = '01/01/1900'
    #convert date string to number
    dt_list  = [datetime.strptime(date, '%d/%m/%Y') for date in dts_list] #note '/' replaced with '_' by csv_read
    #check for id matches
    match_idx = [i for i, j in enumerate(id_list) if j == radar_id]
    #check for multiple matches
    if len(match_idx) == 1:
        dict_idx = match_idx[0]
    else:
        #check list of matches, updates if start_date remains less than odim_date
        for idx in match_idx:
            if radar_dt > dt_list[idx]:
                dict_idx = idx
                
    return dict_idx

def get_field_names():
    """
    Fields name definition.

    Returns:
    ========
        fields_names: array
            Containing [(old key, new key, sig figures), ...]
    """
    fields_names = [('TH', 'total_power'),
                    ('DBZH', 'reflectivity'),
                    ('ZDR', 'differential_reflectivity'),
                    ('PHIDP', 'differential_phase'),
                    ('KDP', 'specific_differential_phase'),
                    ('RHOHV', 'cross_correlation_ratio'),
                    ('VRADH', 'velocity'),
                    ('WRADH', 'spectral_width'),
                    ('SNRH', 'signal_to_noise_ratio')]
    return fields_names

def read_odim(radar_file_name):
    """
    Reads odimh5 volume using pyart and replaces fieldnames as required

    Parameters:
    ===========
        radar_file_name: str
            Full filename to radar volume file 
            
    Returns:
    ========
        radar: Py-ART radar object
            
    """   
    #read radar object
    radar = pyart.aux_io.read_odim_h5(radar_file_name, file_field_names=True)
    #get field names
    fields_names = get_field_names()
    # Parse array old_key, new_key and add least sig digit
    for old_key, new_key in fields_names:
        try:
            radar.add_field(new_key, radar.fields.pop(old_key), replace_existing=True)
        except KeyError:
            continue
    #return radar object
    return radar


def read_csv(csv_ffn, header_line):
    """
    CSV reader used for the radar locations file (comma delimited)
    
    Parameters:
    ===========
        csv_ffn: str
            Full filename to csv file
            
        header_line: int or None
            to use first line of csv as header = 0, use None to use column index
            
    Returns:
    ========
        as_dict: dict
            csv columns are dictionary
    
    """
    df = pandas.read_csv(csv_ffn, header=header_line)
    as_dict = df.to_dict(orient='list')
    return as_dict

def read_cal_file(cal_ffn, zdr_cal=False):
    """
    read Z or ZDR calibration csv file into dictionary
    
    Parameters:
    ===========
        cal_ffn: str
            Full filename to csv file
            
    Returns:
    ========
        cal_dict: dict
            calibration data as dictionary
    
    """
    #load calibration file
    dict_in = read_csv(cal_ffn, None)
    dict_out = {}
    #rename dictonary fields
    dict_out['cal_start'] = np.array([datetime.strptime(str(date), '%Y%m%d').date() for date in dict_in[0]])
    dict_out['cal_end']   = np.array([datetime.strptime(str(date), '%Y%m%d').date() for date in dict_in[1]])
    if zdr_cal: #skip third column which contains information on zdr cal dataset
        dict_out['cal_mean']  = np.array(list(map(float, dict_in[3])))
    else:
        dict_out['cal_mean']  = np.array(list(map(float, dict_in[2])))
    #return
    return dict_out