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

class App(GWSApp):

    @classmethod
    def on_init(cls):
        super().on_init()

        #biota routes
        cls.routes.append(Route('/gaia/home/', homepage) )