$pathToGitRoot = "C:/Users/Alex/Repositories/keras-retinanet"
$pathToSourceRoot = "$($pathToGitRoot)/keras_retinanet/bin"
$pathToTranscript = "$($pathToGitRoot)/Transcripts"

cd $pathToSourceRoot

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)/detailnet_muscima_2018-07-05.txt" -append
python train.py  --epochs 5 --image-min-side 1300 --image-max-side 2000 --backbone detailnet mob_csv ../../data/normalized/muscima/training.csv --val-annotations ../../data/normalized/muscima/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/detailnet_deepscores_2018-07-05.txt" -append
python train.py  --epochs 5 --image-min-side 1300 --image-max-side 2000  --backbone detailnet mob_csv ../../data/normalized/deepscores/training.csv --val-annotations ../../data/normalized/deepscores/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/detailnet_mensural_2018-07-05.txt" -append
python train.py  --epochs 5 --image-min-side 1300 --image-max-side 2000  --backbone detailnet mob_csv ../../data/normalized/mensural/training.csv --val-annotations ../../data/normalized/mensural/validation.csv
Stop-Transcript