import logging
from functools import lru_cache

from lhotse import CutSet, load_manifest
from snowfall.data.asr_datamodule import AsrDataModule


class SogouAsrDataModule(AsrDataModule):
    """
    Aishell ASR data module.
    """
    @lru_cache()
    def train_cuts(self) -> CutSet:
#        logging.info("About to get train cuts")
        print("About to get train cuts")
        return load_manifest(self.args.feature_dir / 'cuts_train.json.gz')

    @lru_cache()
    def valid_cuts(self) -> CutSet:
#        logging.info("About to get valid cuts")
        print("About to get valid cuts")
        return load_manifest(self.args.feature_dir / 'cuts_test8000.json.gz')

    @lru_cache()
    def test_cuts(self) -> CutSet:
        test_sets = ['test8000', 'testIOS', 'not_on_screen', 'testDuiHua']
        cuts = []
        for test_set in test_sets:
#            logging.info("About to get test cuts")
            print("About to get test cuts")
            cuts.append(load_manifest(self.args.feature_dir / f'cuts_{test_set}.json.gz'))
        return cuts


