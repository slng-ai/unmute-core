"""The HTTP package the installed openai client runs on: httpx2 from openai 3, httpx before."""

import importlib

import openai

http = importlib.import_module("httpx2" if int(openai.__version__.split(".")[0]) >= 3 else "httpx")
