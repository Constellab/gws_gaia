import numpy as np
import pandas as pd
from gws_core import (BaseTestCase, ConfigParams, Dataset, DatasetImporter,
                      File, GTest, Settings, TaskRunner, ViewTester)
from gws_core.extra import DataProvider
from gws_gaia import (LMEDesignHelper, LMETrainer)


class TestTrainer(BaseTestCase):

    async def test_lme(self):
        self.print("LMETrainer")
        dataset = Dataset([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [3, 4, 5],
            [3, 4, 5],
            [3, 4, 5],
        ],
            column_names=["glucose", "lactate", "pyruvate"],
        )
        dataset.set_all_row_tags([
            {"time": 1, "dose": 1},
            {"time": "1", "dose": "2"},
            {"time": "2", "dose": "1"},
            {"time": "2", "dose": "2"},
            {"time": "3", "dose": "1"},
            {"time": "3", "dose": "2"},
        ])

        training_design = [{
            "individual": "metabolite",
            "random_effect_structure": ["time:metabolite", "time:metabolite:dose"]
        }]

        df = LMEDesignHelper.create_training_matrix(
            training_set=dataset, training_design=training_design)

        print(df)
        self.assertEqual(df.shape, (18, 4))
        self.assertEqual(df.loc[0, "time"], "1")
        self.assertEqual(df.loc[3, "metabolite"], "glucose")
        self.assertEqual(df.loc[3, "target"], 3)
        self.assertEqual(df.loc[10, "metabolite"], "lactate")

        df = LMEDesignHelper.create_design_matrix(training_matrix=df, training_design=training_design)
        self.assertEqual(df.shape, (18, 2))
        self.assertEqual(df.loc[0, "time:metabolite"], "1_glucose")
        self.assertEqual(df.loc[3, "time:metabolite:dose"], "2_glucose_2")
        print(df)

        # --------------------------------------------------------------------
        # run trainer
        tester = TaskRunner(
            params={'design': training_design},
            inputs={'dataset': dataset},
            task_type=LMETrainer
        )
        outputs = await tester.run()
        trainer_result = outputs['result']
        print(trainer_result)

        # tester = TaskRunner(
        #     inputs={
        #         'learned_model': trainer_result,
        #         'dataset': dataset
        #     },
        #     task_type=LMEPredictor

        # )
        # outputs = await tester.run()
        # predictor_result = outputs['result']

