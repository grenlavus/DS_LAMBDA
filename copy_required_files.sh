#Some initial steps are required when working with this notebook:
#1. You need to enable GPU computing on the notebook;
#2. Clone the repository with our adaptation of the requirements;
#3. You need to install the required dependancies for the project on the environment;
##Cloning Repository:
#executar esses comandos:
#!git clone https://github.com/grenlavus/DS_LAMBDA
#%cd DS_LAMBDA

#!pip install -r requirements-gpu.txt

#from google.colab import drive
#drive.mount('/gdrive',force_remount=True)

#%cd /content/DS_LAMBDA/
#!bash copy_required_files.sh



cd "/gdrive/My Drive/"
echo "Copying Weigths:"
cp "yolov4-obj_best.weights" -P "/content/DS_LAMBDA/data/yolov4.weights"
echo "Copied YoloV4 Weigths"

echo ""

echo "Copying Videos:"

cd "/gdrive/My Drive/unprocessed_videos/"
for file in *
do 
    cp $file -P "/content/DS_LAMBDA/data/video/$file"
    echo "Copied file: $file"
done

echo ""
echo "Saving Model:"

cd /content/DS_LAMBDA/
python save_model.py --model yolov4

echo "Model Saved"