from gatspy.periodic.template_modeler import BaseTemplateModeler

class GatspyTemplateModeler(BaseTemplateModeler):
	"""
	Convenience class for the gatspy BaseTemplateModeler
	"""
	def __init__(self, templates=None, **kwargs):
		self.ftp_templates = None

		if not templates is None:
			if isinstance(templates, dict):
				self.ftp_templates = templates.copy()
			elif isinstance(templates, list):
				self.ftp_templates = { t.template_id : t for t in templates }
			else:
				self.ftp_templates = { 0 : templates }
		BaseTemplateModeler.__init__(self, **kwargs)


	def _template_ids(self):
		return self.ftp_templates.keys()

	def _get_template_by_id(self, template_id):
		assert(template_id in self.ftp_templates)
		t = self.ftp_templates[template_id]

		return t.phase, t.y