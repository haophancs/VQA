org_path=$(pwd)

mkdir vqav2
cd vqav2

wget http://images.cocodataset.org/zips/train2014.zip
unzip -q train2014.zip && rm train2014.zip && mv train2014 train

wget http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip && rm val2014.zip && mv val2014 val

wget http://images.cocodataset.org/zips/test2015.zip
unzip -q test2015.zip && rm test2015.zip && mv test2015 test

mkdir vqa
cd vqa

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
mv v2_mscoco_train2014_annotations.json vqav2_train_annotations.json && rm v2_Annotations_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
mv v2_mscoco_val2014_annotations.json vqav2_val_annotations.json && rm v2_Annotations_Val_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
mv v2_OpenEnded_mscoco_train2014_questions.json vqav2_train_questions.json && rm v2_Questions_Train_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
mv v2_OpenEnded_mscoco_val2014_questions.json vqav2_val_questions.json && rm v2_Questions_Val_mscoco.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
mv v2_OpenEnded_mscoco_test2015_questions.json vqav2_test_questions.json && rm v2_Questions_Test_mscoco.zip

cd $org_path