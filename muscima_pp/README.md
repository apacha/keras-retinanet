# Preparing the dataset

Run the commands in this order
```commandline
python download_muscima_dataset.py
python prepare_muscima_annotations.py
python dataset_splitter.py
```

# Train

```bash
# cd into root
python keras_retinanet/bin/train.py csv data/training.csv data/ClassMapping.csv --val-annotations data/validation.csv
```

