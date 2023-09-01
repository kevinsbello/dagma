{% if obj.display %}
:py:func:`{{ obj.id }} <{{ obj.id }}>`
=========={{ "=" * 2 * (obj.id|length + 2) }}
.. _{{ obj.id }}:
.. py:function:: {{ obj.id }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}

{% for (args, return_annotation) in obj.overloads %}
                 {{ obj.id }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}

{% endfor %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}

   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}

   {% endif %}

{% endif %}
