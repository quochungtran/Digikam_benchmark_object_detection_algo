#Download and unzip annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
mkdir COCOdataset2017
mv annotations COCOdataset2017/