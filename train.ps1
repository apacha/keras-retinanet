$pathToGitRoot = "C:/Users/Alex/Repositories/keras-retinanet"
$pathToSourceRoot = "$($pathToGitRoot)/keras_retinanet/bin"
$pathToTranscript = "$($pathToGitRoot)/Transcripts"

cd $pathToSourceRoot

echo "Appending required paths to temporary PYTHONPATH"
$env:PYTHONPATH = "$($pathToGitRoot);$($pathToSourceRoot)"

Start-Transcript -path "$($pathToTranscript)/train-resnet50-lr-1e-4.txt" -append
python train.py --backbone resnet50 csv ../../data/training.csv ../../data/ClassMapping.csv --val-annotations ../../data/validation.csv
Stop-Transcript

Start-Transcript -path "$($pathToTranscript)/train-densenet121-lr-1e-4.txt" -append
python train.py --backbone densenet121  csv ../../data/training.csv ../../data/ClassMapping.csv --val-annotations ../../data/validation.csv
Stop-Transcript
