import numpy as np
import json
import glob

def load_numpy_data_multifile(filelist):
    allindata = []
    for fname in filelist:
        try:
            item = np.load(
                fname,
                allow_pickle=True,
                encoding="bytes"
            )['arr_0']  # all data
            if not item.shape[0]:
                print("[load_numpy_data_multifile] empy file, skipping: ", fname)
            else:
                allindata.append(item)
        except Exception as e:
            print("[load_numpy_data_multifile] failed to load file: ", fname)
            print("[load_numpy_data_multifile]  ... the error: ")
            print(e)
            print("[load_numpy_data_multifile]  ... skipping this file.")
    data = np.concatenate(allindata) if allindata else np.array(allindata)
    return data


def prepare_data(data, verbose=False, trkdtype='uint8', normcharge=False):
    """

    The function prepares the data in the proper-shape numpy arrays

      data - normally, it is a numpy array of the data for training or prediction

    Data structure:
      [:,0] - calo images
      [:,1] - calo 2 variables  - bgoene (CALO total energy), maxbar (CALO energy of maximum bar)
      [:,2] - truth 4 variables - normally variables that are targeted at the regression optimisation,
      say x_bot, x_top, y_bot, y_top
      [:,3] - rec   4 variables - the same as above, but obtained from the standard BGO rec direction, instead of the
      truth direction
    """

    caloimages = data[:, 0]
    calodata = data[:, 1]
    truthdata = data[:, 2]
    recdata = data[:, 3]

    # get tensor-like shape of the arrays
    caloimages = caloimages.tolist()
    calodata = calodata.tolist()
    truthdata = truthdata.tolist()
    recdata = recdata.tolist()

    caloimages = np.array(caloimages,
                          dtype='float32')  # get a 3-dimensional array, 1st dimension - events, 2,3rd dimensions - image dimensions
    calodata = np.array(calodata)  #
    truthdata = np.array(truthdata)  #
    recdata = np.array(recdata)  #

    # all done
    return {
        'caloimages': caloimages,
        'calodata': calodata,
        'truthdata': truthdata,
        'recdata': recdata,
    }

def get_input_data():
    # get all input data
    data_files = glob.glob('data/tmp_*')
    np.random.seed(1234)
    np.random.shuffle(data_files)
    data = load_numpy_data_multifile(data_files)
    # randomly shuffle the sample
    np.random.shuffle(data)

    # 'prepare' input data (in particular, normalize BGO image, etc.)
    data = prepare_data(data)
    data_target = data['truthdata']
    data_rec = data['recdata']

    return data['caloimages'], data['calodata'], data_target, data_rec
