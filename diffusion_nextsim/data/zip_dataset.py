#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27/11/2023
# Created for diffusion_nextsim
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2023}  {Tobias Sebastian Finn}

# System modules
import logging
import zipfile
import os.path
import warnings

# External modules
from torch.utils.data import Dataset

# Internal modules

main_logger = logging.getLogger(__name__)


class ZipDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            zip_path: str = None,
            extract: bool = True,
            fast: bool = True
    ):
        if fast and os.path.isdir(data_path):
            logging.info(f"Fast checking: {data_path:s} exists")
        elif zip_path is not None:
            correct_extracted = self._check_integrity(data_path, zip_path)
            if not correct_extracted and extract:
                correct_extracted = self._extract_zip(data_path, zip_path)
            if not correct_extracted:
                raise ValueError(
                    "The content in the zip file and in the data path "
                    "are not the same!"
                )
        else:
            warnings.warn(
                f"Can't check the integrity of {data_path:s} "
                f"as no zip path is given!"
            )

    @staticmethod
    def _check_integrity(data_path: str, zip_path: str) -> bool:
        with zipfile.ZipFile(zip_path, mode="r") as f:
            zip_infolist = f.infolist()
        files_exist = []
        for curr_file in zip_infolist:
            file_path = os.path.join(data_path, curr_file.filename)
            files_exist.append(os.path.exists(file_path))
        return all(files_exist)

    def _extract_zip(self, data_path: str, zip_path: str) -> bool:
        with zipfile.ZipFile(zip_path, mode="r") as f:
            main_logger.info(f"Extracting {zip_path:s} into {data_path:s}")
            f.extractall(data_path)
        return self._check_integrity(
            data_path, zip_path
        )
