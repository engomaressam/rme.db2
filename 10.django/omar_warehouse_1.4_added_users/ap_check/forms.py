from django import forms
from .utils import get_distinct_project_names 

class ApCheckForm(forms.Form):
    project_name = forms.ChoiceField(
        choices=[('', '--- Select a Project Name ---')] + 
                [(name, name) for name in get_distinct_project_names()],
        required=True  # Project name is required
    )
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    show_details = forms.BooleanField(required=False, label='Show Detailed List')