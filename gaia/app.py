# Core GWS app module
# This software is the exclusive property of Gencovery SAS. 
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

from gws.settings import Settings
from gws.app import App as GWSApp

from starlette.routing import Route, Mount
from starlette.endpoints import HTTPEndpoint
from starlette.templating import Jinja2Templates

settings = Settings.retrieve()
template_dir = settings.get_template_dir("gaia")
templates = Jinja2Templates(directory=template_dir)

async def homepage(request):
    return templates.TemplateResponse('home.html', {'request': request, 'settings': settings})

class App:
    """
    App class of gaia application

    This App class will dynamically inherits the App classes this application depends on.
    Method on_init() must be overloaded to add new routes.
    """

    routes = []

    @classmethod
    def on_init(cls):
        """
        Initializes the application. 
        
        This method is automatically called after by the constructor.
        """
        # loads base applications' routes
        super().on_init()

        # adds new routes
        cls.routes.append(Route('/gaia/home/', homepage) )