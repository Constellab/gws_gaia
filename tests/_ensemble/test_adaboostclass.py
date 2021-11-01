import os
import asyncio
import time

from gws_core import (Settings, ConfigParams, TaskInputs, TaskTester, BaseTestCase, 
                        ProcessSpec, protocol_decorator, Protocol, IExperiment, IProtocol, File)
from gws_gaia import Dataset, DatasetImporter
from gws_gaia import AdaBoostClassifierTrainer, AdaBoostClassifierPredictor, AdaBoostClassifierTester

@protocol_decorator("AdaBoostClassifierTestProto")
class AdaBoostClassifierTestProto(Protocol):
    def configure_protocol(self, config_params: ConfigParams) -> None:
        p0: ProcessSpec = self.add_process(DatasetImporter, 'p0')
        p1: ProcessSpec = self.add_process(AdaBoostClassifierTrainer, 'p1')
        p2: ProcessSpec = self.add_process(AdaBoostClassifierPredictor, 'p2')
        p3: ProcessSpec = self.add_process(AdaBoostClassifierTester, 'p3')
        self.add_connectors([
            (p0>>'resource', p1<<'dataset'),
            (p0>>'resource', p2<<'dataset'),
            (p1>>'result', p2<<'learned_model'),
            (p1>>'result', p3<<'learned_model'),
            (p0>>'resource', p3<<'dataset')
        ])
        self.add_interface('file', p0, 'file')

class TestTrainer(BaseTestCase):

    async def test_adaboost_proto(self):
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        experiment: IExperiment = IExperiment( AdaBoostClassifierTestProto )
        proto: IProtocol = experiment.get_protocol()
        p0 = proto.get_process("p0")
        p1 = proto.get_process("p1")

        proto.set_input("file", File(path=os.path.join(test_dir, "./iris.csv")))
        p0.set_param("delimiter", ",")
        p0.set_param("header", 0)
        p0.set_param('targets', ['variety'])

        p1.set_param('nb_estimators', 30)

        await experiment.run()
        predictor_result = p1.get_output("result")
        print(predictor_result)

    async def test_adaboost_process(self):
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
        tester = TaskTester(
            params = {'nb_estimators': 30},
            inputs = {'dataset': dataset},
            task_type = AdaBoostClassifierTrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = AdaBoostClassifierPredictor
        )
        outputs = await tester.run()
        predictor_result = outputs['result']

        # run tester
        tester = TaskTester(
            params = {},
            inputs = {
                'dataset': dataset, 
                'learned_model': trainer_result
            },
            task_type = AdaBoostClassifierTester
        )
        outputs = await tester.run()
        tester_result = outputs['result']

        print(trainer_result)
        print(tester_result)
        print(predictor_result)