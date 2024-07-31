import subprocess
from glob import glob
import dataparser as dp
import os
import numpy as np
import argparse
from multiprocessing.dummy import Pool

def imageCSVSave(csv,path_output):
    # convert to compressed CSV
    data = dp.pandasReadImageCSV(csv)
    # reshape to correct size
    if len(data.values.shape)==2:
        data = data.values.reshape((-1,348,464))
    else:
        data = data.values
    # save CSV as compressed NPZ under default key arr_0
    np.savez_compressed(path_output,data)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser(description="Repack compressed CSVs as NPZ and TDMS files. The compressed files are extracted to an intermediary file and deleted once used.")
    # add an option for specifying 7z path
    parser.add_argument("--zip",nargs='?',const="7z",default="7z",type=str,help="Path to 7z executable. If not specified, then it is assumed that 7z is on path")
    # add an option for specifying intermediary path
    parser.add_argument("--ext",nargs='?',const="",default="",type=str,help="Path to intermediary folder where the CSVs are extracted. If not specified then it extracts to local folder")
    # input path
    parser.add_argument("--i","-input",help="Wildcard path to compressed CSV files",required=True)
    # output path
    parser.add_argument("--o","-output",help="Folder path to place the output",required=True)
    # parse arguments
    args = parser.parse_args()

    ## check arguments
    # if the input path is blank
    # default to local
    path_input = args.i
    # check if 7z exists
    zip_exe = args.zip
    # get the rest
    ext_path = "ext_csv"
    
    path_output = args.o
    
    # iterate over the files in the current folder
    # change the path to wherever the 7z files are stored
    for path in glob(path_input):
        # call 7z to extract the file to the specified folder
        # this can take a while
        if os.path.splitext(path)[1] == '.7z':
            subprocess.call([zip_exe,"e",path,"-o"+ext_path])
        # find the file
        csv = None
        for csv in glob(os.path.join(ext_path,"*.csv")):
            fname = os.path.splitext(os.path.basename(csv))[0]
            
            # convert the file to TDMS
            with Pool(3) as pool:
                res_csv = pool.starmap(dp.convertCSVToTDMS,[(csv,os.path.join(path_output,fname+'.tdms'))])
                res_npz = pool.starmap(imageCSVSave,[(csv,os.path.join(path_output,fname+'.npz'))])
        # delete extracted CSV
        # this can be commented out, but given the size of the CSV files
        # can become difficult to manage
        if csv is not None:
            os.remove(csv)
