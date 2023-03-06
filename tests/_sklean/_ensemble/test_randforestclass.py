from gws_core import BaseTestCase, TaskRunner
from gws_gaia import (RandomForestClassifierPredictor,
                      RandomForestClassifierTrainer)
from tests.gws_gaia_test_helper import GWSGaiaTestHelper


class TestTrainer(BaseTestCase):

    def test_process(self):
        self.print("Random forest classifier")
        table = GWSGaiaTestHelper.get_table(index=4, header=0, targets=['target'])

        # run trainer
        tester = TaskRunner(
            params={
                'nb_estimators': 25,
                'training_design': [{'target_name': 'target', 'target_origin': 'row_tag'}],
            },
            inputs={'table': table},
            task_type=RandomForestClassifierTrainer
        )
        outputs = tester.run()
        trainer_result = outputs['result']

        # run predictior
        tester = TaskRunner(
            params={},
            inputs={
                'table': table,
                'learned_model': trainer_result
            },
            task_type=RandomForestClassifierPredictor
        )
        outputs = tester.run()
        predictor_result = outputs['result']

        print(trainer_result)
        print(predictor_result)
