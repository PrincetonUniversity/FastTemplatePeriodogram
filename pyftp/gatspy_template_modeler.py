from __future__ import print_function

from gatspy.periodic.template_modeler import BaseTemplateModeler


class GatspyTemplateModeler(BaseTemplateModeler):
    """
    Convenience class for the gatspy BaseTemplateModeler
    """
    def __init__(self, templates=None, **kwargs):
        assert(not templates is None)

        self.ftp_templates = {}
        self.add_templates(templates)

        #if len(self.ftp_templates) > 0:
        BaseTemplateModeler.__init__(self, **kwargs)

    def _template_ids(self):
        return self.ftp_templates.keys()

    def add_template(self, template, template_id=None):
        if template_id is None:
            if template.template_id is None:
                i = 0
                while i in self.ftp_templates:
                    i+= 1
                self.ftp_templates[i] = template
            else:
                self.ftp_templates[template.template_id] = template
        else:
            self.ftp_templates[template_id] = template
        return self

    def add_templates(self, templates, template_ids=None):

        if isinstance(templates, dict):
            for ID, template in templates.iteritems():
                self.add_template(template, template_id=ID)

        elif isinstance(templates, list):
            if template_ids is None:

                for template in templates:
                    self.add_template(template)
            else:
                for ID, template in zip(template_ids, templates):
                    self.add_template(template, template_id=ID)

        elif not hasattr(templates, '__iter__'):
            self.add_template(templates, template_id=template_ids)

        else:
            raise Exception("did not recognize type of 'templates' passed to add_templates")

        return self

    def _get_template_by_id(self, template_id):
        assert(template_id in self.ftp_templates)
        t = self.ftp_templates[template_id]

        return t.phase, t.y
