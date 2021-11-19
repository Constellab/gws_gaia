import os
import asyncio


from gws_core import (Protocol, Settings, GTest, BaseTestCase, 
                        ConfigParams, ProcessSpec, protocol_decorator,
                        IExperiment)

from gws_gaia.tutorial.tutorial_deepmodel import DeepMoldelTurorialProto


class TestDeepMoldelTurorialProto(BaseTestCase):

    async def test_tutorial_deepmodeler(self):
        self.print("DeepMoldeler tutorial")
        experiment: IExperiment = IExperiment( DeepMoldelTurorialProto )
        await experiment.run()

