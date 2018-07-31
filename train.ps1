$pathToGitRoot = "C:/Users/Alex/Repositories/keras-retinanet"
$pathToSourceRoot = "$($pathToGitRoot)/keras_retinanet/bin"
$pathToTranscript = "$($pathToGitRoot)/Transcripts"

cd $pathToSourceRoot
$number_of_epochs = 500

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)/resnet50_deepscores_2018-07-30.txt" -append
python train.py  --steps 60 --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone resnet50 mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

exit

Start-Transcript -path "$($pathToTranscript)/mobilenet128_muscima_2018-07-05.txt" -append
python train.py  --steps 84  --epochs $number_of_epochs --image-min-side 1000 --image-max-side 1800 --backbone mobilenet128_1 mob_csv ../../data/normalized/muscima/training.csv --val-annotations ../../data/normalized/muscima/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/mobilenet128_deepscores_2018-07-05.txt" -append
python train.py  --steps 60 --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone mobilenet128_1 mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/mobilenet128_mensural_2018-07-05.txt" -append
python train.py  --steps 28  --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone mobilenet128_1 mob_csv ../../data/normalized/mensural/training.csv --val-annotations ../../data/normalized/mensural/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/densenet121_muscima_2018-07-05.txt" -append
python train.py  --steps 84  --epochs $number_of_epochs --image-min-side 800 --image-max-side 1600 --backbone densenet121 mob_csv ../../data/normalized/muscima/training.csv --val-annotations ../../data/normalized/muscima/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/densenet121_deepscores_2018-07-05.txt" -append
python train.py  --steps 60 --epochs $number_of_epochs --image-min-side 1000 --image-max-side 1800  --backbone densenet121 mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/densenet121_mensural_2018-07-05.txt" -append
python train.py  --steps 28  --epochs $number_of_epochs --image-min-side 1000 --image-max-side 1800  --backbone densenet121 mob_csv ../../data/normalized/mensural/training.csv --val-annotations ../../data/normalized/mensural/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/resnet50_muscima_2018-07-05.txt" -append
python train.py  --steps 84 --epochs $number_of_epochs --image-min-side 1000 --image-max-side 1800 --backbone resnet50 mob_csv ../../data/normalized/muscima/training.csv --val-annotations ../../data/normalized/muscima/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/resnet50_deepscores_2018-07-05.txt" -append
python train.py  --steps 60 --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone resnet50 mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/resnet50_mensural_2018-07-05.txt" -append
python train.py  --steps 28  --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone resnet50 mob_csv ../../data/normalized/mensural/training.csv --val-annotations ../../data/normalized/mensural/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/detailnet_muscima_2018-07-05.txt" -append
python train.py  --steps 84  --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000 --backbone detailnet mob_csv ../../data/normalized/muscima/training.csv --val-annotations ../../data/normalized/muscima/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/detailnet_deepscores_2018-07-05.txt" -append
python train.py  --steps 60 --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone detailnet mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/detailnet_mensural_2018-07-05.txt" -append
python train.py  --steps 28  --epochs $number_of_epochs --image-min-side 1300 --image-max-side 2000  --backbone detailnet mob_csv ../../data/normalized/mensural/training.csv --val-annotations ../../data/normalized/mensural/validation.csv
Stop-Transcript