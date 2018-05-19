import os
import shutil

from omrdatasettools.converters.ImageInverter import ImageInverter
from omrdatasettools.downloaders.CvcMuscimaDatasetDownloader import CvcMuscimaDatasetDownloader, CvcMuscimaDataset
from omrdatasettools.downloaders.MuscimaPlusPlusDatasetDownloader import MuscimaPlusPlusDatasetDownloader

if __name__ == "__main__":
    dataset_directory = "../data"
    muscima_pp_raw_dataset_directory = os.path.join(dataset_directory, "muscima_pp_raw")
    muscima_image_directory = os.path.join(dataset_directory, "cvcmuscima_staff_removal")

    print("Deleting dataset directory {0}".format(dataset_directory))
    if os.path.exists(dataset_directory):
        shutil.rmtree(dataset_directory, ignore_errors=True)

    downloader = MuscimaPlusPlusDatasetDownloader(muscima_pp_raw_dataset_directory)
    downloader.download_and_extract_dataset()

    downloader = CvcMuscimaDatasetDownloader(muscima_image_directory, CvcMuscimaDataset.StaffRemoval)
    downloader.download_and_extract_dataset()

    inverter = ImageInverter()
    # We would like to work with black-on-white images instead of white-on-black images
    inverter.invert_images(muscima_image_directory, "*.png")
