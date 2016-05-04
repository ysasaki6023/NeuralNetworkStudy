import cPickle as pickle
import h5py
import numpy as np

fName = "test.pickle"
with open(fName,"rb") as f:
    d = pickle.load(f)

print d.keys()

h5f = h5py.File(fName.replace(".pickle",".h5f"),"w")
h5f.create_group("Info")
h5f.create_group("Original")
h5f.create_group("ZCA_byTrainData")
h5f.create_group("ZCA_byTestData")

h5f.create_dataset("ZCA_byTrainData/data"   ,data=d["data_ZCA_byTrainData"])

h5f.create_dataset("Original/data"     ,data=d["data"])

h5f.create_dataset("ZCA_byTestData/data"    ,data=d["data_ZCA_byTestData"])

h5f.create_dataset("Info/coarse_labels",data=np.array(d["coarse_labels"],dtype=np.int32))
h5f.create_dataset("Info/fine_labels"  ,data=np.array(d["fine_labels"],dtype=np.int32))
#h5f.create_dataset("Info/labels"       ,data=np.array(d["labels"],dtype=np.int32))
dunicode = h5py.special_dtype(vlen=unicode)
h5f.create_dataset("Info/filenames"    ,data=d["filenames"],dtype=dunicode)
h5f.create_dataset("Info/batch_label"  ,data=d["batch_label"],dtype=dunicode)
#h5f.create_dataset("Info/filenames"    ,data=np.array(d["filenames"],dtype=np.unicode))
#h5f.create_dataset("Info/batch_label"  ,data=np.array([d["batch_label"]],dtype=np.unicode))

for i in ['std', 'Uzca', 'C', 'lam', 'eps', 'U', 'V', 'mean']:
    h5f.create_dataset("ZCA_byTrainData/params_%s"%i ,data=d["Params_ZCA_byTrainData"][i])

for i in ['std', 'Uzca', 'C', 'lam', 'eps', 'U', 'V', 'mean']:
    h5f.create_dataset("ZCA_byTestData/params_%s"%i ,data=d["Params_ZCA_byTestData"][i])

h5f.flush()
h5f.close()

