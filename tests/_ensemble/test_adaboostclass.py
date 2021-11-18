import os
import asyncio
import time

from gws_core import (Settings, ConfigParams, TaskInputs, TaskRunner, BaseTestCase, 
                        ProcessSpec, protocol_decorator, Protocol, IExperiment, IProtocol, File)
from gws_gaia import Dataset, DatasetImporter
from gws_gaia import AdaBoostClassifierTrainer, AdaBoostClassifierPredictor

class TestTrainer(BaseTestCase):

    async def test_adaboost_process(self):
        return
        self.print("AdaBoost classifier")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        #import data
        dataset = Dataset.import_from_path(
            File(path=os.path.join(test_dir, "./iris.csv")), 
            ConfigParams({
                "delimiter":",", 
                "header":0, 
                "targets":['variety']
            })
        )
        
        # run trainer
        tester = TaskRunner(
            params = {'nb_estimators': 30},
            inputs = {'dataset': dataset},
            task_type = AdaBoostClassifierTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = AdaBoostClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)