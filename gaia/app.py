# Core GWS app module
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from starlette.routing import Route
from starlette.authentication import requires
from starlette.responses import RedirectResponse

from gws.app import BaseApp

brick = "gaia"

@requires("authenticated")
async def home_page(request):
    return RedirectResponse(url=f'/page/{brick}')

class App(BaseApp):
    """
    App class of Gaia.
    """
    
    @classmethod
    def init(cls):
        """
        Defines current web application routes.

        Routing coventions: 
        
        To prevent route collisions, it is highly recommended to 
        prefix route names of the name of the current brick.
        e.g.: 
            * /<brick name>/home/       -> home page route
            * /<brick name>/settings/   -> setting page route
        """
        # adds new routes
        cls.routes.append(Route(f'/{brick}/', home_page) )