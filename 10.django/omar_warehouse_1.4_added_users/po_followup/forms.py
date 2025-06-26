from django import forms

class PoFollowupForm(forms.Form):
    description = forms.CharField(max_length=200, required=True, 
                                  widget=forms.TextInput(attrs={'id': 'description-search'}))