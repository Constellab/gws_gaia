
import os
import asyncio
from unittest import IsolatedAsyncioTestCase

from gws_core import (ProtocolModel, ExperimentService, ProtocolService, Settings, GTest, protocol_decorator,
                        Protocol, Experiment, ConfigParams, ExperimentService, 
                        BaseTestCase, ProcessSpec, IntParam)
from gws_gaia import Dataset, DatasetLoader
from gws_gaia import AgglomerativeClusteringTrainer

@protocol_decorator("AggclusProtocol")
class AggclusProtocol(Protocol):
    def configure_protocol(self, config_params: ConfigParams) -> None:
        p0: ProcessSpec = self.add_process(DatasetLoader, 'p0')
        p1: ProcessSpec = self.add_process(AgglomerativeClusteringTrainer, 'p1')

        self.add_connectors([
            (p0>>'dataset', p1<<'dataset')
        ])


class TestTrainer(BaseTestCase):
            
    async def test_process(self):
        GTest.print("Agglomerative clustering")
        settings = Settings.retrieve()
        test_dir = settings.get_variable("gws_gaia:testdata_dir")

        proto_model = ProtocolService.create_protocol_model_from_type(AggclusProtocol)
        experiment = ExperimentService.create_experiment_from_protocol_model(proto_model)
        p0 = proto_model.processes["p0"]
        p1 = proto_model.processes["p1"]
        p0.config.set_value("delimiter", ",")
        p0.config.set_value("header", 0)
        p0.config.set_value('targets', ['target1','target2'])
        p0.config.set_value("file_path", os.path.join(test_dir, "./dataset1.csv"))
        p1.config.set_value('nb_clusters', 2)
        p0.config.save()
        p1.config.save()
        experiment.save()
        experiment = await ExperimentService.run_experiment(
            experiment=experiment, user=GTest.user)                


        # experiment = Experiment(AggclusProtocol)
        # protocol = experiment.get_protocol()
        # protocol.get_process("p0").set_param("delimiter", ",")
        # ...
        # experiment.run(user=GTest.user)
        #
        # Avec:
        # experiment -> Interface vers ExperimentModel
        # protocol -> Interface vers ProtocolModel
        
        r1 = p1.outputs.get_resource_model('result')
        print(r1)