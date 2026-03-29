from django import forms


class HomeInputForm(forms.Form):
    context = forms.CharField(widget=forms.Textarea(attrs={"rows": 4}), max_length=4000)
    script = forms.CharField(widget=forms.Textarea(attrs={"rows": 10}), max_length=20000)


class InstructionForm(forms.Form):
    instructions = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={"rows": 3, "placeholder": "Optional instructions for the next refinement"}),
    )
