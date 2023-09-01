{% if obj.display %}
:py:meth:`{{ obj.id }} <{{ obj.id }}>`
=========={{ "=" * 2 * (obj.id|length + 2) }}
.. _{{ obj.id }}:
.. py:method:: {{ obj.id }}({{ obj.args }}){% if obj.return_annotation is not none %} -> {{ obj.return_annotation }}{% endif %}

{% for (args, return_annotation) in obj.overloads %}
               {{ obj.id }}({{ args }}){% if return_annotation is not none %} -> {{ return_annotation }}{% endif %}

{% endfor %}
   {% if obj.properties %}
   {% for property in obj.properties %}
   :{{ property }}:
   {% endfor %}

   {% else %}

   {% endif %}
   {% if obj.docstring %}
   {{ obj.docstring|indent(3) }}

   {% endif %}

{% endif %}
