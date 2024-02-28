import converter
import sac2bin
import datautils
from renamehelper import moveup

if __name__ == '__main__':
    '''
    Generating bird view array from yolo, as the label for sac data
    
    input_video_directory: directory of specific station's video
    output_array_directory: where to save the bird view array
    output_array_directory_simple: where to save the simplified array, [lane1, lane2, lane3] * [total number, big car, small car]
    batch_size : 2^n
    time_offset : there is a gap between the actual time and the time in the file name, we should adjust it 
    '''
    bird_dir = r"F:\35\res_bird64"
    video_dir = "F:/convert/35"
    simple_dir = r"F:\35\res_simple64"
    conv = converter.Converter(input_video_directory=video_dir,
                     output_array_directory=bird_dir,
                     output_array_directory_simple=simple_dir,
                     batch_size=64,
                     time_offset=-87109)
    conv.video2array_directory()
    print("Label created!")

    '''
    Concat [n,z,bird], [1024,1024,bird shape], we take 4s sac data here, so 4s with 250hz should be 1000
    for some reason, we take 1024, to match the number of gpu calculating units
    sac_n, sac_z : sac data on n & z axis
    npydir: directory of label
    
    '''
    sac_n = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHN250.sac"
    sac_z = r"F:\sac4timeshift\1122\231122.000000.EB003035.HHZ250.sac"
    skipped = open("noskiptry.txt", 'w')
    sac2bin_dir = "F:/35/250nz64"
    sac2bin.video2array_directory(labeldir=bird_dir, basedir=sac2bin_dir)
    print("Finish concatenating sac & label!")


    '''
    Transfer the npy file into npz file, nothing else to modify, just make sure the path is set correctly
    npydir: dir of the npy files to be transformed
    npzdir: where to save the npz file
    station: which station this dataset is captured from
    moveup: function that extract all the npz files from each sub-dir
    '''
    npzdir = r"F:\35\250nz64npz"
    STATION = 35
    datautils.to_npz(npydir=sac2bin_dir, npzdir=npzdir, station=STATION)
    moveup(npzdir)
    print("Extraction done!")
    '''
    after extracting the npz file, we have to create the train、test、dev、useless dir manually for the training step
    train: data for training
    dev: data for validation
    test: data for testing
    useless: the very beginning of our video is useless, these data should be deprecated
    '''



