import exiftool, tempfile, subprocess, re, jpeg_ls, os, cv2
import numpy as np
from warnings import warn

"""
    FLIR T540 CSQs

    Compressed SEQ files

    Uses JPEG-LS (Lossless JPEG for images). Header 0x ff d8 ff f7

    Header for frames is
    0x0200hhhhwwww where hhhh is the width as a 2 byte hex string, in little Endian

    e.g. (464,348) -> 0200[d001][5c01]

"""

BATCH_START = '''{
                "$types": {
                    "Cactus.ThermalBatch.Lib.ViewModels.BatchViewModel, Cactus.ThermalBatch.Lib": "1",
                    "Cactus.ThermalBatch.Lib.Jobs.ScaleMinimumTemperatureJob, Cactus.ThermalBatch.Lib": "2",
                    "Cactus.ThermalBatch.Lib.Jobs.ScaleMaximumTemperatureJob, Cactus.ThermalBatch.Lib": "3",
                    "Cactus.ThermalBatch.Lib.Jobs.EmissivityJob, Cactus.ThermalBatch.Lib": "4",
                    "Cactus.Common.Structures.FloatingPointValue, Cactus.Common": "5",
                    "Cactus.ThermalBatch.Lib.Targets.CsvOutputTarget, Cactus.ThermalBatch.Lib": "6",
                    "Cactus.ThermalBatch.Lib.Models.BatchFileModel, Cactus.ThermalBatch.Lib": "7"
                },
            
            "$type": "1",
            "Jobs": [
            '''
BATCH_END = '''],
            "Files": [],
            "LastProcessedImageFiles": [],
            "IsContextMenuOpened": 'false',
            "SelectedItemsCount": 0,
            "IsSummaryEnabled": 'true',
            "AllowMultiSelection": 'true'
        }'''

SET_EM = '''{
            "$type": "4",
            "FloatValue": {
                    "$type": "5",
                    "Value": 0.01,
                    "Minimum": 0.01,
                    "Maximum": 1,
                    "Epsilon": 1.40129846432482e-45
            }
        },'''

CSVOUT = '''{
            "$type": "6",
            "ExportImageData": 'null',
            "Parameters": {
                    "ExportTextAnnotations": "False",
                    "ExportImageParameters": "True"
            }
        },'''

def getMetadata(path):
    with exiftool.ExifToolHelper() as tool:
        return tool.get_metadata(path)

def makeMagic(shape=(464,348)):
    # get hex strings for with and just get number
    temp = hex(shape[0])[2:]
    if len(temp)<4:
        temp = '0' + temp
    # swap bytes around to little endian
    whex = temp[2:4]+temp[0:2]
    temp = hex(shape[1])[2:]
    if len(temp)<4:
        temp = '0' + temp
    # swap bytes around to little endian
    hhex = temp[2:4]+temp[0:2]
    return bytes.fromhex(f"0200{whex}{hhex}")

def makeMagicString(shape=(464,348)):
    # get hex strings for with and just get number
    temp = hex(shape[0])[2:]
    if len(temp)<4:
        temp = '0' + temp
    # swap bytes around to little endian
    whex = temp[2:4]+temp[0:2]
    temp = hex(shape[1])[2:]
    if len(temp)<4:
        temp = '0' + temp
    # swap bytes around to little endian
    hhex = temp[2:4]+temp[0:2]
    return f"0200{whex}{hhex}"

def getMagicLocs(path,shape=(464,348)):
    magic = makeMagic(shape)
    with open(path,'rb') as file:
        rr = file.read(6)
        while True:
            if not rr:
                break
            if magic in rr:
                yield file.tell()-6
            rr = file.read(6)

def getFirstMagicLoc(path,shape=(464,348)):
    magic = makeMagic(shape)
    with open(path,'rb') as file:
        rr = file.read(6)
        while True:
            if not rr:
                break
            if magic in rr:
                return rr,file.tell()
            rr = file.read(6)

def attemptSplitFFF(path,shape=(464,348)):
    magic = makeMagic(shape)
    ct = 0
    # open source file
    with open(path,'rb') as file:
        # open destination file
        cache = bytearray()
        with open(f"seq_{ct}.fff",'wb') as of:
            while True:
                rr = file.read(6)
                # if EOF
                if not rr:
                    ct += 1
                    break
                # if magic in rr
                if magic in rr:
                    of.write(cache)
                    cache.clear()
                else:
                    print(rr)
                    cache.append(rr)
class CSQReader:
    def __init__(self,path):
        # save path
        self._path = path
        # get metadata for file
        self.metadata = getMetadata(path)[0]
        # open file
        self.__fp = open(path,'rb')
        # set chunk size
        self.__csize = int(1e6)
        # chunk idx
        self.__cidx = 0
        # read first __csize bytes
        self.__chunk = self.__fp.read(self.__csize)
        # get image size
        self.__width = self.metadata["FLIR:RawThermalImageWidth"]
        self.__height = self.metadata["FLIR:RawThermalImageHeight"]
        # create magic bytes for file
        self.__magic = makeMagic((self.__width,self.__height))
        # leftover bytes
        self.__lo = b''

    def width(self):
        return self.__width

    def height(self):
        return self.__height

    def _readnext(self):
        # for the current chunk
        # find all matches
        matches = []
        idx = -1
        # add leftovers from previous run
        if self.__lo:
            self.__chunk = self.__lo + self.__chunk
        while True:
            # find next match starting from after the previous find
            idx = self.__chunk.find(self.__magic,idx+1)
            # if magic cannot be found
            # exit
            if idx == -1:
                break
            # append index to list
            matches.append(idx)
        # if magic can't be found
        # return nothing
        if len(matches)==0:
            return
        print(len(matches))
        # iterate over matches
        for mA,mB in zip(matches,matches[1:]):
            yield self.__chunk[mA:mB]
        # save leftover bytes to add to next run
        self.__lo = self.__chunk[mB:]
        self.__chunk = self.__fp.read(self.__csize)

    def readtest(self):
        # for the current chunk
        # find all matches
        matches = []
        idx = -1
        # add leftovers from previous run
        if self.__lo:
            self.__chunk = self.__lo + self.__chunk
        testmagic = bytes.fromhex("ffd8fff7")
        testmagic = self.__magic
        while True:
            # find next match starting from after the previous find
            idx = self.__chunk.find(testmagic,idx+1)
            # if magic cannot be found
            # exit
            if idx == -1:
                break
            # append index to list
            matches.append(idx)
        # if magic can't be found
        # return nothing
        if len(matches)==0:
            return
        print(len(matches))
        # iterate over matches
        for mA,mB in zip(matches,matches[1:]):
            print(len(self.__chunk[mA:mB]),"\n\n")
            print(" ".join([f"{x:02x}" for x in self.__chunk[mA:mB]]))
            try:
                self.decodefile(self.__chunk[mA+len(testmagic):mB])
                print("success!")
            except:
                print("failure")
            input()
        # save leftover bytes to add to next run
        self.__lo = self.__chunk[mB:]
        self.__chunk = self.__fp.read(self.__csize)

    def decodefile(self,e):
        # create temp file to hold binary data
        with tempfile.NamedTemporaryFile() as fp:
        #with open("test","w+b") as fp:
            # write data to file
            fp.write(e)
            fp.flush()
            # get filename
            fname = fp.name
            # get metadata
            metadata = getMetadata(fname)
            # call exiftool to convert binary data to raw thermal sensor readings
            binary = subprocess.check_output(['exiftool', '-b', '-RawThermalImage', fname])
            raw = jpeg_ls.decode(binary)
        # temp file is deleted
        return raw, metadata

    def readNextTemp(self):
        # get next chunk of raw data        
        raw = next(self._readnext())
        # if data is empty
        if not raw:
            print("bork!")
            return
        # create temp file to hold binary data
        with tempfile.NamedTemporaryFile() as fp:
        #with open("test","w+b") as fp:
            # write data to file
            fp.write(raw)
            fp.flush()
            # get filename
            fname = fp.name
            # get metadata
            metadata = getMetadata(fname)
            # call exiftool to convert binary data to raw thermal sensor readings
            binary = subprocess.check_output(['exiftool', '-b', '-RawThermalImage', fname])
            raw = jpeg_ls.decode(np.array(binary))
        # temp file is deleted
        return raw, metadata

def makeEMBatchFile(ems):
    '''
        Make a FLIR batch file for applying a series of emissivity values
        and exporting the CSV
    '''
    batch = BATCH_START
    # append temperature limits
    batch += '{"$type": "2","SelectedTemperatureUnitValue": 0,"Temperature": 150}'
    batch += '{"$type": "3","SelectedTemperatureUnitValue": 0,"Temperature": 1500}'
    # for each emissivity value in the list
    # append a set emissivity job
    for ee in ems:
        # search for where 
        batch += SET_EM.replace('"Value": 0.01',f'"Value": {ee}')
        batch += CSVOUT
    batch += BATCH_END
    return batch.replace(' ','').strip('\\').strip('\n')


class SEQCustomFile:
    """
        Custom class for reading the contents of a SEQ file

        This assumes that the file contains JPEG-LS image data

        On creation is scans and maps out where the image data is located and the user
        can access the specific images via index

        Example 1
        # load image and map the dat
        seq = SEQCustomFile(r"D:\thermal_recording_CSQ_files\2024-visit-1\plate-1-1.seq")
        # print number of frames
        print(seq.numframes())
        # get first frame
        seq[0]

    """
    def __init__(self, path: str, **kwargs) -> None:
        """ 
            Read SEQ file and get data about the file

            Inputs:
                path : Full path to SEQ file
                skip_tags : A flag to skip loading in the tags. Default False
        """
        self._path = path
        self._tags = None
        # useful parameters for calculations
        self.B = None
        self.R1 = None
        self.R2 = None
        self.F = None
        self.O = None
        self.E = None
        self.T_refl = None
        self.Raw_refl = None
        self.fshape = None
        # extract tags using exiftool
        # and update the parameters
        if not kwargs.get("skip_tags", False):
            try:
                with exiftool.ExifToolHelper() as et:
                    self._tags = et.get_metadata(path)[0]

                self.B = self._tags["FLIR:PlanckB"]
                self.R1 = self._tags["FLIR:PlanckR1"]
                self.R2 = self._tags["FLIR:PlanckR2"]
                self.F = self._tags["FLIR:PlanckF"]
                self.O = self._tags["FLIR:PlanckO"]
                self.E = self._tags["FLIR:Emissivity"]
                self.T_refl = self._tags['FLIR:ReflectedApparentTemperature']
                self.Raw_refl = self._tags["FLIR:PlanckR1"]/(self._tags["FLIR:PlanckR2"]*np.exp(self._tags["FLIR:PlanckB"]/self._tags['FLIR:ReflectedApparentTemperature'])-self._tags["FLIR:PlanckF"])-self._tags["FLIR:PlanckO"]
                self.fshape = (self._tags['FLIR:RawThermalImageHeight'], self._tags['FLIR:RawThermalImageWidth'])
            except Exception as e:
                warn(f"Failed to read tags and extract parameters! Reason: {e}")
                
        # list of locations for FF FF FF 00
        self._mlocs = []
        # loc for start of JEG-LS
        self._jlocs = []
        # setup regular expression for searching for FF FF FF 00
        fff_search = re.compile(b"\x46\x46\x46\x00")
        # start + end bytes of a JPEG-LS
        jpeg_start = re.compile(b"\xff\xd8\xff\xf7")
        jpeg_end = re.compile(b"\xFF\xD9")
        # read in file
        data = open(self._path,"rb").read()
        # store starting locations
        self._mlocs.clear()
        # search for FF FF FF 00 markers
        for seg in fff_search.finditer(data):
            self._mlocs.append(seg.span()[0])
            # starting from here
            start = jpeg_start.search(data, seg.span()[1])
            end = jpeg_end.search(data, start.span()[1])
            self._jlocs.append((start.span()[0], end.span()[1]))
        # record the number of frames
        self.__numframes = len(self._jlocs)
        # open the file to avoid re-opening the file
        self._data = open(self._path,"rb")

    # method to get only the flir tags
    def get_flir_tags(self):
        return {k:v for k,v in self._tags.items() if "FLIR:" in k}

    # get the number of frames
    def numframes(self) -> int:
        return self.__numframes

    def __len__(self):
        return self.__numframes

    def __str__(self):
        return f"SEQCustomFile {self._path}, ({self.fshape[0]},{self.fshape[1]}, {self.__numframes})"

    def shape(self):
        return (*self.fshape, self.__numframes)

    # close file pointer
    def close(self):
        self._data.close()

    # reopen file pointer
    def reopen(self):
        self._data = open(self._path,'rb')

    # extract images by index
    # if it fails to decode the image then it returns None
    def __getitem__(self, index: int) -> np.ndarray:
        # check index
        if not isinstance(index, int):
            raise IndexError("Index has to be an integer!")
        if index >= self.__numframes:
            raise IndexError("Index out of bounds!")
        start,end = self._jlocs[index]
        self._data.seek(start)
        # load data
        data_img = self._data.read(abs(end-start))
        data_img_arr = np.array(data_img)
        try:
            # attempt to convert to valid image
            im = jpeg_ls.decode(data_img_arr)
            # update frame shape if not set
            if self.fshape is None:
                self.fshape = im.shape
            return im
        # on encoding error return None
        except RuntimeError:
            blank_arr = np.empty(self.fshape)
            blank_arr.fill(np.nan)
            return blank_arr


    def get_jpeg_data(self, index: int) -> bytearray:
        """
            Extract the raw JPEG data for the specific frame index

            On creation a JPEG lookup table is made to simplify searching for the image data.
            The lookup table is a list of the identified start and end bytes of the JPEG data.
            Invalid images can occur typically due to the identified end of the image being too
            far away from the header (according to the header data) causing it to fail.

            See flir-seq-file-specification.md for mroe details

            Intended as a debug method to investigate invalid or failed images

            Inputs:
                index : Target frame index

            Returns bytearray
        """
        start,end = self._jlocs[index]
        self._data.seek(start)
        data_img = self._data.read(abs(end-start))
        return data_img
    

    def get_max_frame(self) -> np.ndarray:
        """
            Get the frame with the highest max value

            The file is scanned for the valid frame which has the highest raw max value.
            The max value is calulated using np.nanmax

            Invalid frames or those that cause an error are skipped

            Returns identified decoded image
        """
        return max(self, key=lambda x : np.nanmax(x) if not np.isnan(np.nanmax(x)) else -1)


    def get_max_frame_idx(self) -> int:
        """
            Get index of frame with the highest max value

            The file is scanned for the valid frame which has the highest raw max value.
            The max value is calulated using np.nanmax

            Invalid frames or those that cause an error are skipped

            Returns index of frame with highest raw max value
        """
        return int(np.argmax(list(map(lambda x : np.nanmax(x) if not np.isnan(np.nanmax(x)) else -1, self))))

    def apply_each_frame(self, proc) -> list:
        """ 
            Apply the supplied function to each valid frame in the file and store the results in a list

            Example: Finding mean of each frame
                seq = SEQCustomFile("path to file")
                res = seq.apply_each_frame(lambda x : x.mean())

            Inputs:
                proc : Function to apply to each decoded frame

            Returns a list of the results returned by the function
        """
        return list(map(proc, self))
    
    def apply_each_frame_temp(self, proc: callable, E: float, units: str) -> list:
        """ 
            Apply the supplied function to each valid frame in the file when converted to temperature
              and store the results in a list

            Example: Finding mean of each frame
                seq = SEQCustomFile("path to file")
                res = seq.apply_each_frame(lambda x : x.mean())

            The units input states what temperature units the returned frame should be.
            The supported units are as follows:
                K : Kelvin
                F : Farenheit
                C : Celcius

            Anything else is not supported

            Inputs:
                proc : Processing function
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as

            Returns a list of the results returned by the function
        """
        if E is None:
            E = self.E
        if not (units in ["K", "C", "F"]):
            raise ValueError("Received unsupported units string!")
        return list(map(proc, self.tempiter(E, units)))


    # convert raw 16 bit value to radiance
    def _frame2raw(self, frame: np.ndarray, E: float = None) -> np.ndarray:
        """ 
            Convert Raw 16 bit frame to radiance values

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            Used as intermediate step to convert to temperature

            Inputs:
                frame : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None

            Returns converted array
        """
        if E is None:
            E = self.E
        Raw_refl = self.Raw_refl
        # convert 16-bit FLIR RAW to radiance of measured object
        return (frame-(1-E)*Raw_refl)/E
    
    # single function to convert raw 16 bit to temperature
    def frame2temp(self, frame: np.ndarray, E: float = None, units : str = "C") -> np.ndarray:
        """ 
            Convert Raw 16 bit frame to temperature

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            The units input states what temperature units the returned frame should be.
            The supported units are as follows:
                K : Kelvin
                F : Farenheit
                C : Celcius

            Anything else is not supported

            Inputs:
                frame : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as

            Returns array of temperature values in the requested units
        """
        if E is None:
            E = self.E
        if not (units in ["K", "C", "F"]):
            raise ValueError("Received unsupported units string!")
        frame_raw = self._frame2raw(frame, E)
        frame_float = frame_raw.astype("float64")
        # convert radiance values to temperature
        kelvin = self.B/np.log(self.R1/(self.R2*(frame_float+self.O))+self.F)
        # convert to appropriate units
        if units == "K":
            return kelvin
        elif units == "C":
            return kelvin - 273.15
        elif units == "F":
            return (kelvin * 1.8) - 459.67
        
    # iterator for getting the frames as temperature
    def tempiter(self, E: float = None, units : str = "C", start_at: int = 0):
        """
            Alternative iterator that retrieve frames as temperature rather than raw

            Example:
                for frame in SEQCustomFile("path to file").tempiter(units="C"):
                    max_temp = frame.max()

            Inputs:
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as
                start_at : Start at the following frame index 
        """
        # check parameters
        if E is None:
            E = self.E
        if not (units in ["K", "C", "F"]):
            raise ValueError("Received unsupported units string!")
        # iterate over the mapped locations
        for start, end in self._jlocs[start_at:]:
            # go to location
            self._data.seek(start)
            # read data
            data_img = self._data.read(abs(end-start))
            data_img_arr = np.array(data_img)
            # attempt to decode it
            try:
                raw = jpeg_ls.decode(data_img_arr)
                if self.fshape is None:
                    self.fshape = raw.shape
                yield self.frame2temp(raw, E, units)
            # if it failed then return an empty frame
            except RuntimeError:
                blank_arr = np.empty(self.fshape)
                blank_arr.fill(np.nan)
                yield blank_arr


    def get_frame_temp(self, idx: int, E: float = None, units : str = "C"):
        """
            Get specific frame in terms of temeprature

            Inputs:
                idx : Target frame index
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as

            Return frame
        """
        if E is None:
            E = self.E
        if not (units in ["K", "C", "F"]):
            raise ValueError("Received unsupported units string!")
        # get the data range
        start, end = self._jlocs[idx]
        # load the data
        self._data.seek(start)
        data_img = self._data.read(abs(end-start))
        data_img_arr = np.array(data_img)
        # attempt to decode it
        try:
            raw = jpeg_ls.decode(data_img_arr)
            if self.fshape is None:
                self.fshape = raw.shape
            return self.frame2temp(raw, E, units)
        # if it failed then return an empty frame
        except RuntimeError:
            blank_arr = np.empty(self.fshape)
            blank_arr.fill(np.nan)
            return blank_arr


    def export_to_npz(self, opath: str = None, E: float = None, units : str = "C", skip_bad_frames : bool = False):
        """ 
            Load and export the temperature data to a NPZ file at the target location

            NPZ is a compressed numpy array that's a good way to save space and avoid having to go to CSV

            The user can specify what emissivity to use or just use the recording one.
            Allows the user to test different emissivity values.

            The units input states what temperature units the returned frame should be.
            The supported units are as follows:
                K : Kelvin
                F : Farenheit
                C : Celcius

            Anything else is not supported.

            If the flag skip_bad_frames is set to True, the bad frames are not added to the stack.
            If skip_bad_frames is False, a frame of NaNs is added instead. This can be useful for ensuring consistent sizes.

            Inputs:
                opath : Output path for the NPZ file. If None, the path is set to the same folder as the source recording. Default None
                frame : Raw 16 bit values to convert
                E : Emissivity value to use. If None, use value in recording. Default None
                units : Temperature units to return the frame as
                skip_bad_frames : Flag to skip bad frames. Default False
        """
        if E is None:
            E = self.E
        if not (units in ["K", "C", "F"]):
            raise ValueError("Received unsupported units string!")
        # if output path is not set then make one using the original path of the recording
        if opath is None:
            opath = os.path.splitext(os.path.basename(self._path))[0] + ".npz"
        frames = [frame for frame in self.tempiter(E, units)]
        # filter bad values
        if skip_bad_frames:
            frames = list(filter(lambda fr : not np.isnan(fr).all(), frames))
        frames_arr = np.dstack(frames)
        np.savez_compressed(opath, frames_arr)

    # construct a time track for the frames
    def make_time_track(self) -> np.ndarray:
        """ 
            Construct an array of time values useful for plotting

            The number of frames 
        """
        frame_rate = self._tags['FLIR:FrameRate']
        return np.arange(self.__numframes)/float(frame_rate)
    
    # identify the indices of bad frames
    def find_invalid_indices(self) -> list[int]:
        """ 
            Scan the file and find the indices for the INVALID frames

            An frame is invalid/bad if it cannot be decoded correctly

            Returns list of indices corresponding to these bad frames
        """
        invalid_idx = []
        for i in range(self.__numframes):
            try:
                self[i]
            except RuntimeError:
                invalid_idx.append(i)
                continue
        return invalid_idx
    

    # identify the indices of good frames
    def find_valid_indices(self) -> list[int]:
        """ 
            Scan the file and find the indices for the VALID frames

            An frame is valid/good if it can be decoded correctly

            Returns list of indices corresponding to these good frames
        """
        valid_idx = []
        for i in range(self.__numframes):
            try:
                self[i]
                valid_idx.append(i)
            except RuntimeError:
                continue
        return valid_idx
    

    def find_E(self, metric, target: float, accum_func = np.sum, preload_data: bool = False, **kwargs):
        """ 
            Search for an emissivity value that is closest to the target

            The function metric is applied to each temperature frame creating a list.
            The accum_func function is then used to combine these values into a single value
            and the result is returned to scipy.optimize.minimize

            An example of this function is searching for an emissivity value to make the smallest temperature match
            the room temperature

            Example: match room temperature
            # load file
            seq = SEQCustomFile("path to file")
            # search for emissivity so that the min temperature is close to room temperature
            seq.find_E(np.nanmin, 22.5)

            Inputs:
                metric : Function used to evaluate each frame
                target : The target value
                accum_func : Function that accumulates/combines the frame metrics together into a single value
                **kwargs : See scipy.optimize.minimize

            Returns result of minimize
        """
        from scipy.optimize import minimize
        # if preloading the data instead of scanning off disk
        if preload_data:
            data = list(self)
            def frame_metric(E):
                return accum_func([abs(metric(self.frame2temp(frame, E))) for frame in data])
        else:
            def frame_metric(E):
                return accum_func([abs(metric(frame) - target) for frame in self.tempiter(E)])

        return minimize(frame_metric, x0=[self.E,], bounds=[(0.0,0.9)],**kwargs)
    

    # Get the shape of each frame and return the shape
    def get_shapes(self) -> set[tuple[int, int]]:
        """ 
            Get the shape of each frame and store in a list

            Useful as a debugging function for checking size of read in frames

            Returns set of frame shapes
        """
        return set(map(lambda frame: frame.shape, self))
    
    def export_images(self, opath: str):
        fname = os.path.splitext(os.path.basename(opath))[0]
        for i, frame in enumerate(self):
            frame = frame.astype("uint16")
            cv2.imwrite(os.path.join(opath,f"{fname}_frame_{i:06}.png"), frame)


if __name__ == "__main__":
    import numpy as np
    import json
    pass
    #batch = makeEMBatchFile(np.linspace(0.1,1.0))
    #open("batch_processing/mega_test.atb",'wb').write(bytes(batch,'utf-8'))
    #locs = list(getMagicLocs(r"lsbu-doe-stripe-csq/CSQ/lsbu_doe_powder_stripes_0003.csq"))
    attemptSplitFFF(r"lsbu-doe-stripe-csq/CSQ/lsbu_doe_powder_stripes_0003.csq")
