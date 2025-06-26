from django import forms
from .utils import get_distinct_project_names 

class IndirectCostForm(forms.Form):
    project_name = forms.ChoiceField(
        choices=[('', '--- Select a Project Name ---')] + 
                [(name, name) for name in get_distinct_project_names()],
        required=True
    )
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))