
import os
import asyncio


from gws_core import Settings, GTest, BaseTestCase, IExperiment
from gws_gaia.tutorial.tutorial_lda import LDATutorialProto

class TestLDATutorial(BaseTestCase):

    async def test_lda_tutorial(self):
        GTest.print("LDA & PCA tutorial")
        experiment: IExperiment = IExperiment( LDATutorialProto )
        await experiment.run()

                     