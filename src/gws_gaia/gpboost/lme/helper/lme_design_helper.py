# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import pandas as pd
from gws_core import (BadRequestException, BoolParam, ListParam, ParamSet,
                      StrParam, TableTagExtractorHelper)


class LMEDesignHelper:

    @classmethod
    def create_target_param_set(cls):
        return ParamSet({
            'intercept': BoolParam(default_value=True, human_name='Intercept', short_description='Use intercept?'),
            'individual': StrParam(human_name='Individual', short_description='The name of the individual observations'),
            'random_effect_structure': ListParam(human_name='Structure of random effects', short_description="The structure of the (nested-)random effects"),
        }, human_name="Model design", short_description="The design of the model", max_number_of_occurrences=1)

    @classmethod
    def create_training_matrix(cls, training_set, training_design):
        # add covariates
        all_groups = []
        intercept = training_design[0].get("intercept", True)
        individual = training_design[0]["individual"]
        effect_structure = training_design[0]["random_effect_structure"]

        all_groups = []
        for formula in effect_structure:
            groups = formula.split(":")
            all_groups.extend(groups)

        all_groups = list(set(all_groups))
        if individual not in all_groups:
            raise BadRequestException("The individual is not found in found in the randomn effect structure")

        all_groups.remove(individual)

        row_tags = training_set.get_row_tags()
        for key in all_groups:
            if key in row_tags[0]:
                training_set = TableTagExtractorHelper.extract_row_tags(training_set, key, "char")
            else:
                raise BadRequestException(f"The covariate {key} does not exist in the row tags")

        return pd.melt(training_set.get_data(), id_vars=all_groups, var_name=individual, value_name='target')

    @classmethod
    def create_design_matrix(cls, training_matrix, training_design):
        effect_structure = training_design[0]["random_effect_structure"]
        design_matrix = training_matrix.copy()
        for formula in effect_structure:
            groups = formula.split(":")

            for i, colname in enumerate(groups):
                if i == 0:
                    design_matrix[formula] = design_matrix[colname].map(str)
                else:
                    design_matrix[formula] = design_matrix[formula] + "_" + design_matrix[colname].map(str)

        return design_matrix[effect_structure]
