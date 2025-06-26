from django import forms
from .utils import get_distinct_project_names 

class CostDistForm(forms.Form):
    PROJECT_IDENTIFIER_CHOICES = [
        ('project_no', 'Project Number'),
        ('project_name', 'Project Name'),
    ]

    project_identifier_type = forms.ChoiceField(choices=PROJECT_IDENTIFIER_CHOICES, widget=forms.RadioSelect)
    project_identifier = forms.CharField(max_length=200, required=False)  # Make it not required initially
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    show_details = forms.BooleanField(required=False, label='Show Detailed List')

    project_name = forms.ChoiceField(
        choices=[('', '--- Select a Project Name ---')] +  # Add an empty choice for initial selection
                [(name, name) for name in get_distinct_project_names()], 
        required=False
    )

    def clean(self):
        cleaned_data = super().clean()
        project_identifier_type = cleaned_data.get('project_identifier_type')
        project_identifier = cleaned_data.get('project_identifier')
        project_name = cleaned_data.get('project_name')

        if project_identifier_type == 'project_no':
            if not project_identifier:
                self.add_error('project_identifier', 'This field is required when selecting Project Number.')
        elif project_identifier_type == 'project_name':
            if not project_name:
                self.add_error('project_name', 'This field is required when selecting Project Name.')