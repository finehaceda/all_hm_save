from django import forms
from django.core.exceptions import ValidationError
from django.forms import TextInput


class MyForm(forms.Form):
    name = forms.CharField(label='Field 1', max_length=100)
    password = forms.CharField(label='Field 2', max_length=100)
    sex = forms.CharField(label='Field 3', max_length=100)
    # 你可以根据需要添加更多字段

class FileForm(forms.Form):
    file = forms.FileField(required=False)

class EncodeForm(forms.Form):
    file = forms.FileField(required=False)
    # file = forms.FileField(upload_to='uploads/',required=False)
    # data = forms.CharField(widget=forms.Textarea, label='请输入要编码的信息：', required=True)
    sequence_length = forms.IntegerField(widget=TextInput(attrs={"type": "range"}))
    mingc = forms.IntegerField(widget=TextInput(attrs={"type": "range"}))
    maxgc = forms.IntegerField(widget=TextInput(attrs={"type": "range"}))
    method1 = forms.ChoiceField(choices=[
        ('fountain', '喷泉码'), ('yyc', 'YYC'), ('derrick', 'derrick'), ('hedges', 'hedges'), ('polar', '极化码')],
                                label='选择编码方法：', required=True)
    method2 = forms.ChoiceField(choices=[
        ('fountain', '喷泉码'), ('yyc', 'YYC'), ('derrick', 'derrick'), ('hedges', 'hedges'), ('polar', '极化码')],
        label='选择编码方法：', required=True)

class EncodeHiddenForm(forms.Form):
    # 隐藏字段
    hidden_method1 = forms.CharField(widget=forms.HiddenInput, required=False)
    hidden_method2 = forms.CharField(widget=forms.HiddenInput, required=False)
    hidden_mingc = forms.IntegerField(widget=forms.HiddenInput, required=False)
    hidden_maxgc = forms.IntegerField(widget=forms.HiddenInput, required=False)
    hidden_sequence_length = forms.IntegerField(widget=forms.HiddenInput, required=False)
    info_density1 = forms.FloatField(widget=forms.HiddenInput, required=False)
    encode_time1 = forms.CharField(widget=forms.HiddenInput, required=False)
    sequence_number1 = forms.IntegerField(widget=forms.HiddenInput, required=False)
    index_length1 = forms.IntegerField(widget=forms.HiddenInput, required=False)
    info_density2 = forms.FloatField(widget=forms.HiddenInput, required=False)
    encode_time2 = forms.CharField(widget=forms.HiddenInput, required=False)
    sequence_number2 = forms.IntegerField(widget=forms.HiddenInput, required=False)
    index_length2 = forms.IntegerField(widget=forms.HiddenInput, required=False)

    def clean_hidden_method1(self):
        name = self.cleaned_data.get('name')
        if not name:
            raise ValidationError("encode method is required.")
        return name