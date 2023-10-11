from pathlib import Path

from fastapi import Response
from jinja2 import Environment, FileSystemLoader


class Templater:
    def __init__(self):
        self.templates = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates/")
        )

    def render_template(self, template_name: str, **kwargs):
        template = self.templates.get_template(template_name)
        return template.render(**kwargs)

    def get_connection_twiml(self, call_id: str, base_url: str):
        return Response(
            self.render_template("connect_call.xml", base_url=base_url, id=call_id),
            media_type="application/xml",
        )

    def get_bypass_twiml(self, transfer_to: str):
        return Response(
            self.render_template("bypass_transfer.xml", transfer_to=transfer_to),
            media_type="application/xml",
        )
