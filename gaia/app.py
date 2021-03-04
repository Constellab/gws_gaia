# Core GWS app module
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
from gws.settings import Settings

class API:
    
    @staticmethod
    async def run_lda_pca(request):
        from gaia._tuto.tutorial import lda_pca_experiment

        settings = Settings.retrieve()
        test_dir = settings.get_dir("gaia:testdata_dir")
        data_file = os.path.join(test_dir, "./iris.csv")

        e = lda_pca_experiment(data_file, delimiter=",", header=0, target=['variety'], ncomp=2)
        await e.run()
        e.save()

        return e.view().as_json()
    
