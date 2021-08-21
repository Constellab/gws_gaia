# LICENSE
# This software is the exclusive property of Gencovery SAS.
# The use and distribution of this software is prohibited without the prior consent of Gencovery SAS.
# About us: https://gencovery.com

import os
import sys
from subprocess import check_call
import setuptools
from setuptools.command.install import install

NAME = "gws_gaia"
VERSION = "0.1.0"
DESCRIPTION = "Gencovery Artificial Intelligence and Analytics (GAIA)"
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class InstallHook(install):
    """Installation hooks (for production mode)."""

    def _run_install(self, what ):
        cwd = os.path.join(self.install_lib, NAME)
        script_path = os.path.join(cwd, ".hooks", f"{what}-install.sh")
        if os.path.exists(script_path):
            check_call([ "bash", script_path ], cwd=cwd)
        script_path = os.path.join(cwd, ".hooks", f"{what}-install.py")
        if os.path.exists(script_path):
            check_call([ sys.executable, script_path ], cwd=cwd)

    def _run_pre_install(self):
        self.announce("Running pre-install hook ...")
        self._run_install("pre")
        self.announce("Done!")

    def _run_post_install(self):
        self.announce("Running post-install hook ...")
        self._run_install("post")
        self.announce("Done!")

    def run(self):
        self._run_pre_install()
        install.run(self)
        self._run_post_install()
